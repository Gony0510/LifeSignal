from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import serial

try:
    from ForMov.config import CSV_COLUMNS, DEFAULT_DATA_PATH, SAMPLE_INTERVAL_SEC
    from ForMov.preprocessing import normalize_label
    from ForMov.serial_common import (
        CAPTURE_START_COMMAND,
        CAPTURE_STOP_COMMAND,
        AmbiguousPortError,
        PortUnavailableError,
        SampleRateLimiter,
        open_sensor_serial,
        parse_sample_line,
        resolve_port,
    )
except ModuleNotFoundError:
    from config import CSV_COLUMNS, DEFAULT_DATA_PATH, SAMPLE_INTERVAL_SEC  # type: ignore
    from preprocessing import normalize_label  # type: ignore
    from serial_common import (  # type: ignore
        CAPTURE_START_COMMAND,
        CAPTURE_STOP_COMMAND,
        AmbiguousPortError,
        PortUnavailableError,
        SampleRateLimiter,
        open_sensor_serial,
        parse_sample_line,
        resolve_port,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_session_is_new(path: Path, session_id: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                "CSV 열 구성이 영상용 스키마와 다릅니다. 다른 출력 파일을 사용해주세요."
            )
        if any(str(row.get("session_id", "")) == session_id for row in reader):
            raise ValueError(
                f"이미 존재하는 세션입니다: {session_id}. 새 세션 이름을 사용해주세요."
            )


def make_row(
    sample: dict[str, object], *, label: str, session_id: str
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "label": label,
        "timestamp": utc_now(),
        "sample_millis": sample.get("sample_millis"),
        "sensor_id": "c4001",
        "room": sample.get("room"),
        "location": sample.get("location"),
        "status": sample.get("status") is True,
        "target_energy": sample.get("target_energy"),
        "source_session_id": session_id,
        "is_augmented": False,
        "augmentation_id": "",
    }


def collect(args: argparse.Namespace) -> None:
    label = normalize_label(args.label)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_session_is_new(destination, args.session)
    write_header = not destination.exists() or destination.stat().st_size == 0
    target_count = max(1, int(round(args.duration / args.sample_interval)))
    print(
        f"영상용 수집 준비: label={label}, session={args.session}, "
        f"목표={target_count}개({args.duration:.1f}초)"
    )
    print(f"CSV 저장 위치: {destination}")

    count = 0
    preferred_port: str | None = None
    first_deadline = time.monotonic() + args.startup_timeout
    received_any = False
    reconnect_notice = False

    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            stream.flush()

        while count < target_count:
            connection: serial.Serial | None = None
            try:
                if not received_any and time.monotonic() >= first_deadline:
                    raise TimeoutError("C4001 원시 측정값이 도착하지 않았습니다.")
                port = resolve_port(args.port, preferred_port)
                connection = open_sensor_serial(port, args.baud, args.boot_wait)
                preferred_port = port
                if not reconnect_notice:
                    print(f"C4001 시리얼 연결: {port}")
                received_on_connection = False
                connection_started = time.monotonic()
                last_sample_at: float | None = None
                last_start_command_at = 0.0
                limiter = SampleRateLimiter(args.sample_interval)

                while count < target_count:
                    now = time.monotonic()
                    if not received_any and now >= first_deadline:
                        raise TimeoutError("C4001 원시 측정값이 도착하지 않았습니다.")
                    if (
                        received_any
                        and not received_on_connection
                        and now - connection_started >= args.startup_timeout
                    ):
                        raise serial.SerialException("재연결 후 샘플이 도착하지 않습니다.")
                    if last_sample_at is not None and now - last_sample_at >= args.sample_timeout:
                        raise serial.SerialException("원시 샘플 수신이 중단되었습니다.")
                    if not received_on_connection and now - last_start_command_at >= 1.0:
                        connection.write(CAPTURE_START_COMMAND)
                        connection.flush()
                        last_start_command_at = now

                    raw_line = connection.readline()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        sample = parse_sample_line(line)
                    except ValueError as exc:
                        print(f"손상된 원시 샘플 무시: {exc}")
                        continue
                    if sample is None:
                        print(line)
                        continue

                    received_on_connection = True
                    last_sample_at = time.monotonic()
                    if not received_any:
                        received_any = True
                        print("C4001 원시 데이터 수집을 시작합니다.")
                    elif reconnect_notice:
                        reconnect_notice = False
                        print("C4001 연결 복구: 같은 세션 수집을 이어갑니다.")
                    if not limiter.accept(last_sample_at):
                        continue

                    writer.writerow(make_row(sample, label=label, session_id=args.session))
                    stream.flush()
                    count += 1
                    if count == 1 or count % 25 == 0 or count == target_count:
                        print(f"수집 진행: {count}/{target_count}개")

            except (PortUnavailableError, serial.SerialException, OSError) as exc:
                if not received_any and time.monotonic() >= first_deadline:
                    raise TimeoutError("C4001 센서 또는 포트를 확인해주세요.") from exc
                if not reconnect_notice:
                    print("C4001 연결 끊김: 센서를 다시 연결하면 자동으로 이어서 수집합니다.")
                reconnect_notice = True
                time.sleep(args.reconnect_interval)
            finally:
                if connection is not None:
                    try:
                        if connection.is_open:
                            connection.write(CAPTURE_STOP_COMMAND)
                            connection.flush()
                            connection.close()
                    except (serial.SerialException, OSError):
                        pass
    print(f"수집 완료: {destination} ({count}개)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상용 C4001 원시 데이터를 0.2초 간격으로 CSV에 저장"
    )
    parser.add_argument("--port", help="XIAO ESP32-S3 USB 포트(생략 시 자동 탐색)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--label",
        required=True,
        help="sleeping, evacuating, fallen, no_signal 중 하나",
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL_SEC)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--sample-timeout", type=float, default=3.0)
    parser.add_argument("--reconnect-interval", type=float, default=2.0)
    parser.add_argument("--boot-wait", type=float, default=2.0)
    parser.add_argument("--output", default=str(DEFAULT_DATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("duration", "sample_interval", "startup_timeout", "sample_timeout"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name} 값은 0보다 커야 합니다.")
    if args.reconnect_interval <= 0 or args.boot_wait < 0:
        raise SystemExit("재연결 간격은 0보다 커야 하고 부팅 대기는 0 이상이어야 합니다.")
    try:
        collect(args)
    except (AmbiguousPortError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"영상용 C4001 수집 실패: {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n영상용 데이터 수집을 중단했습니다.")

