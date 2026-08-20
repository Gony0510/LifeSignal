from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import serial

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ForMov.config import (  # noqa: E402
    SAMPLE_INTERVAL_SEC,
    VPR100_CSV_COLUMNS,
    VPR100_DEFAULT_DATA_PATH,
)
from ForMov.preprocessing import normalize_label  # noqa: E402
from serial_bridge import (  # noqa: E402
    BridgeConfig,
    LatestMessageBuffer,
    RadarInputParser,
    open_serial,
    resolve_serial_port,
)


def ensure_session_is_new(path: Path, session_id: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != VPR100_CSV_COLUMNS:
            raise ValueError(
                "CSV 열 구성이 영상용 V-PR100 스키마와 다릅니다. "
                "다른 출력 파일을 사용해주세요."
            )
        if any(str(row.get("session_id", "")) == session_id for row in reader):
            raise ValueError(
                f"이미 존재하는 세션입니다: {session_id}. 새 세션 이름을 사용해주세요."
            )


def make_row(
    sample: dict[str, object], *, label: str, session_id: str
) -> dict[str, object]:
    score = sample.get("presence_score")
    try:
        normalized_score = max(0, int(float(score)))
    except (TypeError, ValueError):
        normalized_score = 0
    distance = sample.get("distance_mm")
    try:
        normalized_distance: int | None = max(0, int(float(distance)))
    except (TypeError, ValueError):
        normalized_distance = None
    return {
        "session_id": session_id,
        "label": label,
        "timestamp": sample.get("timestamp"),
        "sensor": "V-PR100",
        "room": sample.get("room"),
        "location": sample.get("location"),
        "status": sample.get("status") is True,
        "presence_score": normalized_score,
        "distance_mm": normalized_distance,
        "source_session_id": session_id,
        "is_augmented": False,
        "augmentation_id": "",
    }


def make_bridge_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        room=args.room,
        location=args.location,
        presence_threshold=args.presence_threshold,
        presence_off_threshold=args.presence_off_threshold,
        confirmation_count=args.confirmation_count,
        absence_timeout=args.absence_timeout,
        motion_hold=args.motion_hold,
    )


def collect(args: argparse.Namespace) -> None:
    label = normalize_label(args.label)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_session_is_new(destination, args.session)
    write_header = not destination.exists() or destination.stat().st_size == 0
    target_count = max(1, int(round(args.duration / args.sample_interval)))
    print(
        f"영상용 V-PR100 수집 준비: label={label}, session={args.session}, "
        f"목표={target_count}개({args.duration:.1f}초)"
    )
    print(f"CSV 저장 위치: {destination}")

    config = make_bridge_config(args)
    count = 0
    preferred_port = args.port
    first_deadline = time.monotonic() + args.startup_timeout
    received_any = False
    reconnect_notice = False

    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=VPR100_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            stream.flush()

        while count < target_count:
            connection: serial.Serial | None = None
            try:
                if not received_any and time.monotonic() >= first_deadline:
                    raise TimeoutError("V-PR100 측정값이 도착하지 않았습니다.")
                try:
                    port = resolve_serial_port(preferred_port)
                except SystemExit as exc:
                    raise serial.SerialException(str(exc)) from exc
                connection = open_serial(port, args.baud)
                if connection is None:
                    raise serial.SerialException("V-PR100 시리얼 포트를 열 수 없습니다.")
                preferred_port = port
                if not reconnect_notice:
                    print(f"V-PR100 시리얼 연결: {port} ({args.protocol})")

                parser = RadarInputParser(config, args.protocol)
                publisher = LatestMessageBuffer(args.sample_interval)
                connection_started = time.monotonic()
                received_on_connection = False
                last_message_at: float | None = None

                while count < target_count:
                    now = time.monotonic()
                    if not received_any and now >= first_deadline:
                        raise TimeoutError("V-PR100 측정값이 도착하지 않았습니다.")
                    if (
                        received_any
                        and not received_on_connection
                        and now - connection_started >= args.startup_timeout
                    ):
                        raise serial.SerialException("재연결 후 측정값이 도착하지 않습니다.")
                    if last_message_at is not None and now - last_message_at >= args.sample_timeout:
                        raise serial.SerialException("V-PR100 측정값 수신이 중단되었습니다.")

                    waiting = connection.in_waiting
                    if waiting > 0:
                        messages = [
                            message
                            for message in parser.feed(connection.read(waiting), now)
                            if message.get("presence_score") is not None
                        ]
                        if messages:
                            publisher.push(messages)
                            received_on_connection = True
                            last_message_at = now
                            if not received_any:
                                received_any = True
                                print("V-PR100 데이터 수집을 시작합니다.")
                            elif reconnect_notice:
                                reconnect_notice = False
                                print("V-PR100 연결 복구: 같은 세션 수집을 이어갑니다.")

                    due_message = publisher.pop_due(now)
                    if due_message is not None:
                        writer.writerow(
                            make_row(due_message, label=label, session_id=args.session)
                        )
                        stream.flush()
                        count += 1
                        print(
                            f"\r수집 진행: {count}/{target_count}개",
                            end="",
                            flush=True,
                        )
                    time.sleep(0.01)

            except TimeoutError:
                raise
            except (serial.SerialException, OSError) as exc:
                if not received_any and time.monotonic() >= first_deadline:
                    raise TimeoutError("V-PR100 센서 또는 포트를 확인해주세요.") from exc
                if not reconnect_notice:
                    print(
                        "\nV-PR100 연결 끊김: 센서를 다시 연결하면 "
                        "자동으로 이어서 수집합니다."
                    )
                reconnect_notice = True
                time.sleep(args.reconnect_interval)
            finally:
                if connection is not None and connection.is_open:
                    try:
                        connection.close()
                    except (serial.SerialException, OSError):
                        pass
    print(f"\n수집 완료: {destination} ({count}개)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상용 V-PR100 데이터를 0.2초 간격으로 CSV에 직접 저장"
    )
    parser.add_argument(
        "--port",
        help="PL2303 USB-RS232 포트(생략 시 자동 탐색)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--protocol", choices=["auto", "text", "binary"], default="binary")
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
    parser.add_argument("--output", default=str(VPR100_DEFAULT_DATA_PATH))
    parser.add_argument("--room", type=int, default=401)
    parser.add_argument("--location", default="거실A")
    parser.add_argument("--presence-threshold", type=int, default=1000)
    parser.add_argument("--presence-off-threshold", type=int, default=700)
    parser.add_argument("--confirmation-count", type=int, default=2)
    parser.add_argument("--absence-timeout", type=float, default=5.0)
    parser.add_argument("--motion-hold", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "duration",
        "sample_interval",
        "startup_timeout",
        "sample_timeout",
        "reconnect_interval",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name} 값은 0보다 커야 합니다.")
    try:
        collect(args)
    except (TimeoutError, ValueError) as exc:
        raise SystemExit(f"영상용 V-PR100 수집 실패: {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n영상용 V-PR100 데이터 수집을 중단했습니다.")
