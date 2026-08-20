from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serial_bridge import open_serial, resolve_serial_port  # noqa: E402


STX = 0x02
ETX = 0x03
CONFIG_PACKET_SIZE = 17
MIN_START_DECIMETERS = 2
MAX_END_DECIMETERS = 70


def is_config_packet(packet: bytes) -> bool:
    if len(packet) != CONFIG_PACKET_SIZE:
        return False
    if packet[0] != STX or packet[-1] != ETX:
        return False
    module_id, start, end, profile = packet[1:5]
    return (
        1 <= module_id <= 99
        and MIN_START_DECIMETERS <= start < end <= MAX_END_DECIMETERS
        and 1 <= profile <= 5
    )


def find_config_packet(data: bytes) -> bytes | None:
    for start_index in range(len(data) - CONFIG_PACKET_SIZE, -1, -1):
        packet = data[start_index : start_index + CONFIG_PACKET_SIZE]
        if is_config_packet(packet):
            return packet
    return None


def distance_to_decimeters(distance_m: float) -> int:
    decimeters = int(round(distance_m * 10))
    if abs(distance_m * 10 - decimeters) > 1e-6:
        raise ValueError("탐지 거리는 0.1m 단위로 지정해야 합니다.")
    if not MIN_START_DECIMETERS < decimeters <= MAX_END_DECIMETERS:
        raise ValueError("종료 거리는 0.3m 이상 7.0m 이하이어야 합니다.")
    return decimeters


def update_end_distance(packet: bytes, end_distance_m: float) -> bytes:
    if not is_config_packet(packet):
        raise ValueError("V-PR100 설정 패킷 형식이 올바르지 않습니다.")
    end_decimeters = distance_to_decimeters(end_distance_m)
    if end_decimeters <= packet[2]:
        raise ValueError(
            f"종료 거리는 현재 시작 거리 {packet[2] / 10:.1f}m보다 커야 합니다."
        )
    updated = bytearray(packet)
    updated[3] = end_decimeters
    return bytes(updated)


def describe_packet(packet: bytes) -> str:
    threshold = int.from_bytes(packet[5:7], byteorder="big") / 100
    return (
        f"시작={packet[2] / 10:.1f}m, 종료={packet[3] / 10:.1f}m, "
        f"Profile={packet[4]}, Threshold={threshold:.2f}"
    )


def send_ascii(connection: serial.Serial, command: str) -> None:
    connection.write(command.encode("ascii"))
    connection.flush()


def read_config_packet(connection: serial.Serial, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    received = bytearray()
    while time.monotonic() < deadline:
        waiting = connection.in_waiting
        chunk = connection.read(waiting if waiting > 0 else 1)
        if chunk:
            received.extend(chunk)
            packet = find_config_packet(bytes(received))
            if packet is not None:
                return packet
        else:
            time.sleep(0.01)
    preview = bytes(received[-64:]).hex(" ") or "수신 없음"
    raise TimeoutError(
        "V-PR100 설정 패킷을 받지 못했습니다. "
        f"마지막 수신 데이터: {preview}"
    )


def request_current_config(
    connection: serial.Serial,
    *,
    timeout: float,
    command_delay: float,
) -> bytes:
    connection.reset_input_buffer()
    send_ascii(connection, "info")
    time.sleep(command_delay)
    return read_config_packet(connection, timeout)


def configure(args: argparse.Namespace) -> None:
    port = resolve_serial_port(args.port)
    connection = open_serial(port, args.baud)
    if connection is None:
        raise serial.SerialException("V-PR100 시리얼 포트를 열 수 없습니다.")

    entered_command_mode = False
    try:
        print(f"V-PR100 연결: {port}")
        connection.reset_input_buffer()
        send_ascii(connection, "command")
        entered_command_mode = True
        time.sleep(args.command_delay)

        current = request_current_config(
            connection,
            timeout=args.timeout,
            command_delay=args.command_delay,
        )
        print(f"현재 설정: {describe_packet(current)}")
        updated = update_end_distance(current, args.end_distance_m)

        if updated == current:
            print("종료 거리가 이미 요청한 값으로 설정되어 있습니다.")
            return

        connection.reset_input_buffer()
        connection.write(updated)
        connection.flush()
        time.sleep(args.command_delay)

        verified = request_current_config(
            connection,
            timeout=args.timeout,
            command_delay=args.command_delay,
        )
        if verified != updated:
            raise RuntimeError(
                "설정 검증에 실패했습니다. 센서가 반환한 설정이 전송값과 다릅니다.\n"
                f"전송: {updated.hex(' ')}\n"
                f"수신: {verified.hex(' ')}"
            )
        print(f"변경 완료: {describe_packet(verified)}")
        print("나머지 센서 설정은 그대로 유지했습니다.")
    finally:
        if entered_command_mode:
            try:
                send_ascii(connection, "detect")
                time.sleep(args.command_delay)
                print("V-PR100 탐지 모드를 다시 시작했습니다.")
            except (serial.SerialException, OSError):
                print("경고: detect 명령 전송에 실패했습니다. 센서 전원을 다시 연결해주세요.")
        if connection.is_open:
            connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상용 V-PR100의 현재 설정을 보존하며 종료 거리만 변경"
    )
    parser.add_argument(
        "--port",
        help="PL2303 USB-RS232 포트(생략 시 자동 탐색)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--end-distance-m", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--command-delay", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0 or args.command_delay < 0:
        raise SystemExit("timeout은 0보다 커야 하고 command-delay는 0 이상이어야 합니다.")
    try:
        configure(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"V-PR100 종료 거리 설정 실패: {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nV-PR100 설정을 중단했습니다.")
