from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import serial
import serial.tools.list_ports

try:
    from AI.preprocessing import normalize_label
except ModuleNotFoundError:
    from preprocessing import normalize_label  # type: ignore


SAMPLE_PREFIX = "@C4001_SAMPLE "
CAPTURE_START_COMMAND = b"C4001_CAPTURE_START\n"
CAPTURE_STOP_COMMAND = b"C4001_CAPTURE_STOP\n"
CSV_COLUMNS = (
    "session_id",
    "label",
    "timestamp",
    "sample_millis",
    "sensor_id",
    "room",
    "location",
    "status",
    "target_energy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_sample_line(
    line: str,
    *,
    label: str,
    session_id: str,
    timestamp: str | None = None,
) -> dict[str, object] | None:
    if not line.startswith(SAMPLE_PREFIX):
        return None

    try:
        data = json.loads(line[len(SAMPLE_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("C4001 원시 측정값 JSON이 올바르지 않습니다.") from exc
    if not isinstance(data, dict) or data.get("type") != "c4001_sample":
        raise ValueError("C4001 원시 측정값 형식이 아닙니다.")

    row: dict[str, object] = {
        "session_id": session_id,
        "label": label,
        "timestamp": timestamp or utc_now(),
        "sample_millis": data.get("sample_millis"),
        "sensor_id": "c4001",
        "room": data.get("room"),
        "location": data.get("location"),
        "status": data.get("status"),
        "target_energy": data.get("target_energy"),
    }
    return row


def available_ports() -> list[str]:
    return [port.device for port in serial.tools.list_ports.comports()]


class PortUnavailableError(RuntimeError):
    pass


class AmbiguousPortError(RuntimeError):
    pass


class ActiveCaptureTimer:
    """연결된 구간에서 실제 샘플이 도착한 시간만 누적합니다."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.accumulated = 0.0
        self.connection_first_sample_at: float | None = None
        self.connection_last_sample_at: float | None = None

    def observe_sample(self, now: float) -> bool:
        if self.connection_first_sample_at is None:
            self.connection_first_sample_at = now
        self.connection_last_sample_at = now
        return self.elapsed >= self.duration

    @property
    def elapsed(self) -> float:
        current_span = 0.0
        if (
            self.connection_first_sample_at is not None
            and self.connection_last_sample_at is not None
        ):
            current_span = (
                self.connection_last_sample_at
                - self.connection_first_sample_at
            )
        return self.accumulated + current_span

    def pause(self) -> None:
        if (
            self.connection_first_sample_at is not None
            and self.connection_last_sample_at is not None
        ):
            self.accumulated += (
                self.connection_last_sample_at
                - self.connection_first_sample_at
            )
        self.connection_first_sample_at = None
        self.connection_last_sample_at = None


def resolve_port(
    requested_port: str | None,
    preferred_port: str | None = None,
) -> str:
    if requested_port:
        return requested_port

    candidates = [
        port
        for port in available_ports()
        if port.startswith("/dev/cu.usbmodem")
        or port.startswith("/dev/cu.usbserial")
        or port.startswith("/dev/ttyACM")
        or port.startswith("/dev/ttyUSB")
    ]
    if preferred_port in candidates:
        return preferred_port
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise PortUnavailableError(
            "C4001 ESP32 시리얼 포트를 찾지 못했습니다. "
            "--port로 포트를 지정해주세요."
        )
    raise AmbiguousPortError(
        "USB 시리얼 포트가 여러 개입니다. --port로 하나를 지정해주세요:\n- "
        + "\n- ".join(candidates)
    )


def collect(args: argparse.Namespace) -> None:
    label = normalize_label(args.label)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0

    print(
        f"수집 준비: label={label}, session={args.session}, "
        f"duration={args.duration:.1f}초"
    )
    print(f"CSV 저장 위치: {destination}")

    count = 0
    progress = ActiveCaptureTimer(args.duration)
    initial_deadline = time.monotonic() + args.startup_timeout
    received_any_sample = False
    preferred_port: str | None = None
    waiting_for_reconnect = False

    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            stream.flush()

        while progress.elapsed < args.duration:
            sensor_serial: serial.Serial | None = None
            try:
                if not received_any_sample and time.monotonic() >= initial_deadline:
                    raise TimeoutError(
                        "C4001 원시 측정값을 받지 못했습니다. "
                        "펌웨어 업로드와 포트 연결을 확인해주세요."
                    )

                port = resolve_port(args.port, preferred_port)
                if not waiting_for_reconnect:
                    print(f"C4001 시리얼 포트 연결 중: {port}")
                sensor_serial = serial.Serial(port, args.baud, timeout=0.2)
                preferred_port = port
                time.sleep(args.boot_wait)
                sensor_serial.reset_input_buffer()

                connection_started_at = time.monotonic()
                last_start_command_at = 0.0
                received_on_connection = False

                while progress.elapsed < args.duration:
                    now = time.monotonic()
                    if (
                        not received_any_sample
                        and now >= initial_deadline
                    ):
                        raise TimeoutError(
                            "C4001 원시 측정값을 받지 못했습니다. "
                            "펌웨어 업로드와 포트 연결을 확인해주세요."
                        )
                    if (
                        not received_on_connection
                        and received_any_sample
                        and now - connection_started_at >= args.startup_timeout
                    ):
                        raise serial.SerialException(
                            "재연결 후 원시 측정값이 도착하지 않습니다."
                        )
                    if (
                        received_on_connection
                        and progress.connection_last_sample_at is not None
                        and now - progress.connection_last_sample_at
                        >= args.sample_timeout
                    ):
                        raise serial.SerialException(
                            "원시 측정값 수신이 중단되었습니다."
                        )
                    if (
                        not received_on_connection
                        and now - last_start_command_at >= 1.0
                    ):
                        sensor_serial.write(CAPTURE_START_COMMAND)
                        sensor_serial.flush()
                        last_start_command_at = now

                    raw_line = sensor_serial.readline()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    try:
                        row = parse_sample_line(
                            line,
                            label=label,
                            session_id=args.session,
                        )
                    except ValueError as exc:
                        print(f"원시 측정값 무시: {exc}")
                        continue

                    if row is None:
                        # 기존 아두이노 상태 메시지만 사용자 화면에 표시합니다.
                        print(line)
                        continue

                    first_sample_after_connect = not received_on_connection
                    received_on_connection = True
                    if not received_any_sample:
                        received_any_sample = True
                        print("C4001 원시 데이터 수집을 시작합니다.")
                    elif first_sample_after_connect and waiting_for_reconnect:
                        print("C4001 연결 복구: 기존 세션 수집을 재개합니다.")
                    waiting_for_reconnect = False

                    completed = progress.observe_sample(time.monotonic())
                    writer.writerow(row)
                    stream.flush()
                    count += 1
                    if completed:
                        break

            except AmbiguousPortError:
                raise
            except TimeoutError:
                raise
            except (PortUnavailableError, serial.SerialException, OSError) as exc:
                if not received_any_sample and time.monotonic() >= initial_deadline:
                    raise TimeoutError(
                        "C4001 원시 측정값을 받지 못했습니다. "
                        "펌웨어 업로드와 포트 연결을 확인해주세요."
                    ) from exc
                if not waiting_for_reconnect:
                    state = "재연결" if received_any_sample else "연결"
                    print(
                        f"C4001 연결 끊김 또는 포트 없음: {state} 대기 중..."
                    )
                waiting_for_reconnect = True
                time.sleep(args.reconnect_interval)
            finally:
                progress.pause()
                if sensor_serial is not None:
                    try:
                        if sensor_serial.is_open:
                            sensor_serial.write(CAPTURE_STOP_COMMAND)
                            sensor_serial.flush()
                            sensor_serial.close()
                    except (serial.SerialException, OSError):
                        pass

    print(f"수집 완료: {destination} ({count}개)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C4001 USB 시리얼 원시 데이터를 CSV로 저장"
    )
    parser.add_argument(
        "--port",
        help="ESP32 USB 시리얼 포트(생략 시 macOS/Linux 포트 자동 탐색)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--label",
        required=True,
        help="human(사람), empty(빈 공간), dog(개)",
    )
    parser.add_argument("--session", required=True, help="독립 측정 세션 이름")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument(
        "--sample-timeout",
        type=float,
        default=3.0,
        help="측정값이 없을 때 연결 끊김으로 판단할 시간(초)",
    )
    parser.add_argument(
        "--reconnect-interval",
        type=float,
        default=2.0,
        help="USB 재연결 시도 간격(초)",
    )
    parser.add_argument("--boot-wait", type=float, default=2.0)
    parser.add_argument("--output", default="AI/data/c4001_samples.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("수집 시간은 0보다 커야 합니다.")
    if args.startup_timeout <= 0:
        raise SystemExit("시작 대기시간은 0보다 커야 합니다.")
    if args.sample_timeout <= 0:
        raise SystemExit("측정값 대기시간은 0보다 커야 합니다.")
    if args.reconnect_interval <= 0:
        raise SystemExit("재연결 시도 간격은 0보다 커야 합니다.")
    if args.boot_wait < 0:
        raise SystemExit("부팅 대기시간은 0 이상이어야 합니다.")
    try:
        collect(args)
    except (AmbiguousPortError, serial.SerialException, TimeoutError) as exc:
        raise SystemExit(f"C4001 데이터 수집 실패: {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nC4001 데이터 수집을 중단했습니다.")
