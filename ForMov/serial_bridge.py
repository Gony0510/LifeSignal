from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

import serial
import websockets

try:
    from ForMov.config import SAMPLE_INTERVAL_SEC
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
    from config import SAMPLE_INTERVAL_SEC  # type: ignore
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


def make_payload(sample: dict[str, object]) -> dict[str, object]:
    return {
        "type": "radar_data",
        "sensor": "C4001-ForMov",
        "room": sample.get("room", 401),
        "location": sample.get("location", "거실A"),
        "status": sample.get("status") is True,
        "target_energy": sample.get("target_energy", 0),
        "sample_millis": sample.get("sample_millis"),
        "timestamp": utc_now(),
        "reason": "c4001_raw_sample",
    }


async def bridge_once(
    args: argparse.Namespace,
    *,
    preferred_port: str | None,
) -> str:
    port = resolve_port(args.port, preferred_port)
    connection: serial.Serial | None = None
    try:
        connection = await asyncio.to_thread(
            open_sensor_serial, port, args.baud, args.boot_wait
        )
        print(f"C4001 시리얼 연결 성공: {port}")
        async with websockets.connect(args.ws_url) as websocket:
            print(f"영상용 서버 연결 성공: {args.ws_url}")
            limiter = SampleRateLimiter(args.sample_interval)
            last_start_command_at = 0.0
            last_sample_at: float | None = None
            received_sample = False
            started_at = time.monotonic()

            while True:
                now = time.monotonic()
                if not received_sample and now - started_at >= args.startup_timeout:
                    raise TimeoutError("C4001 원시 샘플이 도착하지 않았습니다.")
                if last_sample_at is not None and now - last_sample_at >= args.sample_timeout:
                    raise serial.SerialException("C4001 원시 샘플 수신이 중단되었습니다.")
                if not received_sample and now - last_start_command_at >= 1.0:
                    connection.write(CAPTURE_START_COMMAND)
                    connection.flush()
                    last_start_command_at = now

                raw_line = await asyncio.to_thread(connection.readline)
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
                received_sample = True
                last_sample_at = time.monotonic()
                if not limiter.accept(last_sample_at):
                    continue
                payload = make_payload(sample)
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                print(
                    "센서 상태 전송: "
                    f"감지={payload['status']} 에너지={payload['target_energy']}"
                )
    finally:
        if connection is not None:
            try:
                if connection.is_open:
                    connection.write(CAPTURE_STOP_COMMAND)
                    connection.flush()
                    connection.close()
            except (serial.SerialException, OSError):
                pass
    return port


async def run(args: argparse.Namespace) -> None:
    preferred_port: str | None = None
    while True:
        try:
            preferred_port = await bridge_once(args, preferred_port=preferred_port)
        except AmbiguousPortError:
            raise
        except (PortUnavailableError, TimeoutError, serial.SerialException, OSError) as exc:
            print(f"센서 연결 대기: {exc}")
        except (ConnectionError, websockets.WebSocketException) as exc:
            print(f"영상용 서버 재연결 대기: {exc}")
        await asyncio.sleep(args.reconnect_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C4001 원시 샘플을 영상용 AI 서버로 0.2초마다 전송"
    )
    parser.add_argument("--port", help="XIAO ESP32-S3 USB 포트(생략 시 자동 탐색)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8881")
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL_SEC)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--sample-timeout", type=float, default=3.0)
    parser.add_argument("--reconnect-interval", type=float, default=2.0)
    parser.add_argument("--boot-wait", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.sample_interval,
        args.startup_timeout,
        args.sample_timeout,
        args.reconnect_interval,
    ) <= 0:
        raise SystemExit("시간 관련 옵션은 0보다 커야 합니다.")
    if args.boot_wait < 0:
        raise SystemExit("부팅 대기시간은 0 이상이어야 합니다.")
    try:
        asyncio.run(run(args))
    except AmbiguousPortError as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        print("\n영상용 C4001 브리지를 종료했습니다.")


if __name__ == "__main__":
    main()

