from __future__ import annotations

import json
import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports


SAMPLE_PREFIX = "@C4001_SAMPLE "
CAPTURE_START_COMMAND = b"C4001_CAPTURE_START\n"
CAPTURE_STOP_COMMAND = b"C4001_CAPTURE_STOP\n"


class PortUnavailableError(RuntimeError):
    pass


class AmbiguousPortError(RuntimeError):
    pass


def available_ports() -> list[str]:
    return [port.device for port in serial.tools.list_ports.comports()]


def resolve_port(requested_port: str | None, preferred_port: str | None = None) -> str:
    if requested_port:
        return requested_port
    candidates = [
        port
        for port in available_ports()
        if port.startswith("/dev/cu.usbmodem")
        or port.startswith("/dev/cu.usbserial")
        or port.startswith("/dev/cu.wchusbserial")
    ]
    if preferred_port in candidates:
        return str(preferred_port)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise PortUnavailableError(
            "C4001 XIAO ESP32-S3 시리얼 포트를 찾지 못했습니다. "
            "--port로 포트를 지정해주세요."
        )
    raise AmbiguousPortError(
        "USB 시리얼 포트가 여러 개입니다. --port로 하나를 지정해주세요:\n- "
        + "\n- ".join(candidates)
    )


def parse_sample_line(line: str) -> dict[str, object] | None:
    if not line.startswith(SAMPLE_PREFIX):
        return None
    try:
        data = json.loads(line[len(SAMPLE_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("C4001 원시 측정값 JSON이 올바르지 않습니다.") from exc
    if not isinstance(data, dict) or data.get("type") != "c4001_sample":
        raise ValueError("C4001 원시 측정값 형식이 아닙니다.")
    try:
        energy = float(data.get("target_energy"))
    except (TypeError, ValueError) as exc:
        raise ValueError("target_energy가 숫자가 아닙니다.") from exc
    data["target_energy"] = int(round(max(0.0, energy)))
    data["status"] = data.get("status") is True
    return data


@dataclass
class SampleRateLimiter:
    interval: float
    last_accepted_at: float | None = None

    def accept(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if self.last_accepted_at is None:
            self.last_accepted_at = current
            return True
        if current - self.last_accepted_at + 1e-6 < self.interval:
            return False
        self.last_accepted_at = current
        return True


def open_sensor_serial(port: str, baud: int, boot_wait: float) -> serial.Serial:
    connection = serial.Serial(port, baud, timeout=0.2)
    if boot_wait:
        time.sleep(boot_wait)
    connection.reset_input_buffer()
    return connection

