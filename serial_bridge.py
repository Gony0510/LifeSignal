import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic

import serial
import serial.tools.list_ports
import websockets


DEFAULT_WS_URL = "ws://127.0.0.1:8881"
DEFAULT_PUBLISH_INTERVAL = 0.2

PRESENCE_PATTERN = re.compile(
    r"Presence\s+score:\s*(\d+)\s*,\s*Distance:\s*(\d+)",
    re.IGNORECASE,
)
MOTION_PATTERN = re.compile(r"\bMotion\b", re.IGNORECASE)
NO_MOTION_PATTERN = re.compile(r"\bNo\s+Motion\b", re.IGNORECASE)
ABSENCE_PATTERN = re.compile(
    r"\b(?:No\s+Presence|Absence|No\s+Target)\b",
    re.IGNORECASE,
)
LOG_PREFIX_PATTERN = re.compile(r"^\s*<[^>]+>\s*app:\s*", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BridgeConfig:
    room: int
    location: str
    presence_threshold: int
    presence_off_threshold: int
    confirmation_count: int
    absence_timeout: float
    motion_hold: float


class RadarParser:
    """V-PR100 텍스트 로그를 줄 단위의 센서 상태로 변환합니다."""

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.buffer = ""
        self.status = False
        self.motion = False
        self.presence_score: int | None = None
        self.distance_mm: int | None = None
        self.last_detection_at: float | None = None
        self.last_motion_at: float | None = None
        self.timeout_sent = False
        self.confirmation_streak = 0

    def feed(self, raw_bytes: bytes, now: float | None = None) -> list[dict]:
        now = monotonic() if now is None else now
        self.buffer += raw_bytes.decode("utf-8", errors="replace").replace("\r", "\n")
        messages: list[dict] = []

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            message = self.parse_line(line, now)
            if message is not None:
                messages.append(message)

        return messages

    def flush(self, now: float | None = None) -> dict | None:
        """줄바꿈 없이 끝난 데모/테스트 입력을 처리합니다."""
        if not self.buffer.strip():
            self.buffer = ""
            return None
        line, self.buffer = self.buffer, ""
        return self.parse_line(line, monotonic() if now is None else now)

    def parse_line(self, raw_line: str, now: float | None = None) -> dict | None:
        now = monotonic() if now is None else now
        line = LOG_PREFIX_PATTERN.sub("", raw_line).strip()
        if not line:
            return None

        recognized = False
        reason = "measurement"

        if ABSENCE_PATTERN.search(line):
            recognized = True
            self.status = False
            self.motion = False
            self.presence_score = 0
            self.distance_mm = None
            self.timeout_sent = True
            self.confirmation_streak = 0
            reason = "sensor_absence"
        elif NO_MOTION_PATTERN.search(line):
            recognized = True
            self.motion = False
            reason = "no_motion"
        elif MOTION_PATTERN.search(line):
            recognized = True
            self.motion = True
            self.last_motion_at = now
            reason = "motion"

        match = PRESENCE_PATTERN.search(line)
        if match:
            recognized = True
            self.presence_score = int(match.group(1))
            self.distance_mm = int(match.group(2))

            if self.status:
                if self.presence_score < self.config.presence_off_threshold:
                    self.status = False
                    self.confirmation_streak = 0
                    self.timeout_sent = True
                    reason = "below_off_threshold"
                else:
                    self.last_detection_at = now
                    self.timeout_sent = False
            elif self.presence_score >= self.config.presence_threshold:
                self.confirmation_streak += 1
                if self.confirmation_streak >= self.config.confirmation_count:
                    self.status = True
                    self.last_detection_at = now
                    self.timeout_sent = False
                    reason = "presence_confirmed"
                else:
                    reason = "pending_confirmation"
            else:
                self.confirmation_streak = 0
                self.status = False
                self.timeout_sent = True
                reason = "below_threshold"

            if self.status:
                self.last_detection_at = now
                self.timeout_sent = False

        if not recognized:
            return None

        self._expire_motion(now)
        return self._payload(raw_line.strip(), reason)

    def timeout_message(self, now: float | None = None) -> dict | None:
        now = monotonic() if now is None else now
        self._expire_motion(now)
        if (
            self.status
            and self.last_detection_at is not None
            and now - self.last_detection_at >= self.config.absence_timeout
            and not self.timeout_sent
        ):
            self.status = False
            self.motion = False
            self.timeout_sent = True
            self.confirmation_streak = 0
            return self._payload("", "timeout")
        return None

    def _expire_motion(self, now: float) -> None:
        if (
            self.motion
            and self.last_motion_at is not None
            and now - self.last_motion_at >= self.config.motion_hold
        ):
            self.motion = False

    def _payload(self, raw_text: str, reason: str) -> dict:
        return {
            "type": "radar_data",
            "sensor": "V-PR100",
            "room": self.config.room,
            "location": self.config.location,
            "status": self.status,
            "motion": self.motion,
            "presence_score": self.presence_score,
            "distance_mm": self.distance_mm,
            "distance_m": (
                round(self.distance_mm / 1000, 3)
                if self.distance_mm is not None
                else None
            ),
            "timestamp": utc_now(),
            "reason": reason,
            "raw_text": raw_text,
        }


class BinaryRadarParser:
    """V-PR100 Presence Serial의 8/12바이트 패킷을 구조화합니다."""

    LEGACY_PACKET_SIZE = 8
    DISTANCE_PACKET_SIZE = 12

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.buffer = bytearray()

    def feed(self, raw_bytes: bytes, now: float | None = None) -> list[dict]:
        del now
        self.buffer.extend(raw_bytes)
        messages: list[dict] = []

        while self.buffer:
            try:
                start_index = self.buffer.index(0x02)
            except ValueError:
                self.buffer.clear()
                break
            if start_index:
                del self.buffer[:start_index]
            if len(self.buffer) < self.LEGACY_PACKET_SIZE:
                break

            module_id = self.buffer[1]
            detect = self.buffer[2]
            if not 1 <= module_id <= 99 or detect not in {0, 1}:
                del self.buffer[0]
                continue

            distance_mm: int | None = None
            if (
                len(self.buffer) >= self.DISTANCE_PACKET_SIZE
                and self.buffer[self.DISTANCE_PACKET_SIZE - 1] == 0x03
            ):
                packet_size = self.DISTANCE_PACKET_SIZE
                packet = bytes(self.buffer[:packet_size])
                distance_mm = int.from_bytes(
                    packet[7:11], byteorder="big", signed=False
                )
            elif self.buffer[self.LEGACY_PACKET_SIZE - 1] == 0x03:
                packet_size = self.LEGACY_PACKET_SIZE
                packet = bytes(self.buffer[:packet_size])
            elif len(self.buffer) < self.DISTANCE_PACKET_SIZE:
                # 확장 패킷이 나머지 바이트를 수신할 때까지 기다립니다.
                break
            else:
                del self.buffer[0]
                continue

            del self.buffer[:packet_size]
            score = int.from_bytes(packet[3:7], byteorder="big", signed=False)
            status = detect == 1
            messages.append(
                {
                    "type": "radar_data",
                    "sensor": "V-PR100",
                    "room": self.config.room,
                    "location": self.config.location,
                    "status": status,
                    "motion": None,
                    "motion_available": False,
                    "presence_score": score,
                    "distance_mm": distance_mm,
                    "distance_m": (
                        round(distance_mm / 1000, 3)
                        if distance_mm is not None
                        else None
                    ),
                    "timestamp": utc_now(),
                    "reason": (
                        "serial_presence" if status else "serial_absence"
                    ),
                    "module_id": module_id,
                    "raw_hex": packet.hex(" "),
                    "raw_text": "",
                }
            )

        return messages

    def flush(self, now: float | None = None) -> dict | None:
        del now
        return None

    def timeout_message(self, now: float | None = None) -> dict | None:
        del now
        return None


class RadarInputParser:
    """USB 텍스트와 공식 RS232 바이너리를 자동 판별합니다."""

    def __init__(self, config: BridgeConfig, protocol: str = "auto") -> None:
        if protocol not in {"auto", "text", "binary"}:
            raise ValueError(f"지원하지 않는 프로토콜입니다: {protocol}")
        self.protocol = protocol
        self.selected_protocol = None if protocol == "auto" else protocol
        self.text_parser = RadarParser(config)
        self.binary_parser = BinaryRadarParser(config)

    def feed(self, raw_bytes: bytes, now: float | None = None) -> list[dict]:
        if self.selected_protocol == "text":
            return self.text_parser.feed(raw_bytes, now)
        if self.selected_protocol == "binary":
            return self.binary_parser.feed(raw_bytes, now)

        binary_messages = self.binary_parser.feed(raw_bytes, now)
        text_messages = self.text_parser.feed(raw_bytes, now)
        if binary_messages:
            self.selected_protocol = "binary"
            print("시리얼 입력 자동 판별: 공식 RS232 바이너리")
            return binary_messages
        if text_messages:
            self.selected_protocol = "text"
            print("시리얼 입력 자동 판별: USB 텍스트 로그")
            return text_messages
        return []

    def flush(self, now: float | None = None) -> dict | None:
        if self.selected_protocol == "binary":
            return None
        return self.text_parser.flush(now)

    def timeout_message(self, now: float | None = None) -> dict | None:
        if self.selected_protocol in {None, "binary"}:
            return None
        return self.text_parser.timeout_message(now)


def print_available_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("인식된 시리얼 포트가 없습니다.")
        return
    print("현재 인식된 시리얼 포트:")
    for port in ports:
        print(
            f"- device={port.device}, description={port.description}, "
            f"vid={port.vid}, pid={port.pid}"
        )


def resolve_serial_port(requested_port: str | None) -> str:
    """명시된 포트 또는 V-PR100용 USB-RS232 포트를 반환합니다."""
    if requested_port:
        return requested_port

    candidates: list[str] = []
    for port in serial.tools.list_ports.comports():
        device = port.device or ""
        description = port.description or ""
        search_text = f"{device} {description}".lower()
        is_pl2303 = (
            getattr(port, "vid", None) == 0x067B
            or "pl2303" in search_text
        )
        is_usb_serial = (
            "usbserial" in search_text
            or "usb-serial" in search_text
            or "usb serial" in search_text
            or device.startswith("/dev/ttyUSB")
        )
        if is_pl2303 or is_usb_serial:
            candidates.append(device)

    if len(candidates) == 1:
        print(f"V-PR100 시리얼 포트 자동 선택: {candidates[0]}")
        return candidates[0]

    print_available_ports()
    if not candidates:
        raise SystemExit(
            "V-PR100용 USB-RS232 포트를 찾지 못했습니다. "
            "--port 옵션으로 포트를 지정해 주세요."
        )
    raise SystemExit(
        "V-PR100 후보 포트가 여러 개입니다. --port 옵션으로 지정해 주세요: "
        + ", ".join(candidates)
    )


def open_serial(port: str, baud_rate: int) -> serial.Serial | None:
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"시리얼 포트 연결 성공: {port}")
        return ser
    except serial.SerialException as exc:
        print(f"시리얼 포트 연결 실패: {exc}")
        print_available_ports()
        return None


async def send_messages(websocket, messages: list[dict]) -> None:
    for message in messages:
        await websocket.send(json.dumps(message, ensure_ascii=False))
        distance_text = (
            f"{message['distance_mm']}mm"
            if message["distance_mm"] is not None
            else "-"
        )
        print(
            "센서 상태 전송: "
            f"감지={message['status']} 움직임={message['motion']} "
            f"점수={message['presence_score']} 거리={distance_text} "
            f"판정={message['reason']}"
        )


class LatestMessageBuffer:
    """센서 입력은 계속 소비하면서 지정 주기마다 최신 상태만 꺼냅니다."""

    def __init__(self, interval: float, now: float | None = None) -> None:
        self.interval = interval
        start = monotonic() if now is None else now
        self.last_published_at = start - interval
        self.pending: dict | None = None

    def push(self, messages: list[dict]) -> None:
        if messages:
            self.pending = messages[-1]

    def pop_due(self, now: float | None = None) -> dict | None:
        now = monotonic() if now is None else now
        if (
            self.pending is None
            or now - self.last_published_at < self.interval
        ):
            return None
        message, self.pending = self.pending, None
        self.last_published_at = now
        return message


async def serial_bridge(args: argparse.Namespace, parser: RadarInputParser) -> None:
    port = resolve_serial_port(args.port)
    ser = open_serial(port, args.baud)
    if ser is None:
        return

    try:
        async with websockets.connect(args.ws_url) as websocket:
            print(f"웹소켓 서버 연결 성공: {args.ws_url}")
            print(f"시리얼 입력 프로토콜: {args.protocol}")
            publisher = LatestMessageBuffer(args.publish_interval)
            print(f"상태 표시/전송 주기: {args.publish_interval:.2f}초")
            while True:
                waiting = ser.in_waiting
                if waiting > 0:
                    publisher.push(parser.feed(ser.read(waiting)))

                timeout_message = parser.timeout_message()
                if timeout_message is not None:
                    publisher.push([timeout_message])

                due_message = publisher.pop_due()
                if due_message is not None:
                    await send_messages(websocket, [due_message])
                await asyncio.sleep(0.01)
    except (ConnectionRefusedError, OSError) as exc:
        print(f"웹소켓 서버에 연결할 수 없습니다: {args.ws_url} ({exc})")
    except websockets.ConnectionClosed as exc:
        print(f"웹소켓 연결 종료: {exc}")
    except serial.SerialException as exc:
        print(f"시리얼 통신 오류: {exc}")
    finally:
        if ser.is_open:
            ser.close()
            print("시리얼 포트를 닫았습니다.")


async def serial_bridge_forever(
    args: argparse.Namespace,
    config: BridgeConfig,
) -> None:
    """포트나 서버가 일시적으로 끊겨도 브리지를 계속 재시작합니다."""
    while True:
        parser = RadarInputParser(config, args.protocol)
        try:
            await serial_bridge(args, parser)
        except SystemExit as exc:
            print(f"V-PR100 연결 대기: {exc}")
        print(f"{args.reconnect_interval:.1f}초 후 V-PR100 연결을 다시 시도합니다.")
        await asyncio.sleep(args.reconnect_interval)


async def demo_bridge(args: argparse.Namespace, parser: RadarInputParser) -> None:
    samples = [
        "<info> app: Motion\n",
        "<info> app: Presence score: 120, Distance: 4200\n",
        "<info> app: Presence score: 2338, Distance: 600\n",
        "<info> app: Presence score: 2200, Distance: 650\n",
        "<info> app: Presence score: 1980, Distance: 1100\n",
        "<info> app: Presence score: 1640, Distance: 1750\n",
        "<info> app: Presence score: 1210, Distance: 2400\n",
        "<info> app: No Presence\n",
    ]
    async with websockets.connect(args.ws_url) as websocket:
        print(f"데모 모드: {args.publish_interval:.2f}초 간격으로 전송합니다.")
        while True:
            for sample in samples:
                await send_messages(websocket, parser.feed(sample.encode("utf-8")))
                await asyncio.sleep(args.publish_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-PR100 시리얼-WebSocket 브리지")
    parser.add_argument(
        "--port",
        help="시리얼 포트(생략 시 PL2303/usbserial 포트를 자동 탐색)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--room", type=int, default=401)
    parser.add_argument("--location", default="거실A")
    parser.add_argument("--presence-threshold", type=int, default=1000)
    parser.add_argument("--presence-off-threshold", type=int, default=700)
    parser.add_argument("--confirmation-count", type=int, default=2)
    parser.add_argument("--absence-timeout", type=float, default=5.0)
    parser.add_argument("--motion-hold", type=float, default=2.0)
    parser.add_argument(
        "--publish-interval",
        type=float,
        default=DEFAULT_PUBLISH_INTERVAL,
        help="서버로 보낼 최신 센서 상태의 주기(초, 기본 0.2)",
    )
    parser.add_argument(
        "--reconnect-interval",
        type=float,
        default=2.0,
        help="시리얼 또는 서버 연결 종료 후 재시도 간격(초)",
    )
    parser.add_argument(
        "--protocol",
        choices=["auto", "text", "binary"],
        default="auto",
        help="auto: USB 텍스트/RS232 바이너리 자동 판별",
    )
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.presence_off_threshold > args.presence_threshold:
        raise SystemExit("해제 임계값은 감지 임계값보다 작거나 같아야 합니다.")
    if args.confirmation_count < 1:
        raise SystemExit("연속 확인 횟수는 1 이상이어야 합니다.")
    if args.publish_interval <= 0:
        raise SystemExit("전송 주기는 0보다 커야 합니다.")
    if args.reconnect_interval <= 0:
        raise SystemExit("재연결 주기는 0보다 커야 합니다.")

    config = BridgeConfig(
        room=args.room,
        location=args.location,
        presence_threshold=args.presence_threshold,
        presence_off_threshold=args.presence_off_threshold,
        confirmation_count=args.confirmation_count,
        absence_timeout=args.absence_timeout,
        motion_hold=args.motion_hold,
    )
    if args.demo:
        parser = RadarInputParser(config, args.protocol)
        await demo_bridge(args, parser)
    else:
        await serial_bridge_forever(args, config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n브리지 프로그램을 종료합니다.")
