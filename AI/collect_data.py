from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import websockets

try:
    from AI.preprocessing import normalize_label
except ModuleNotFoundError:
    from preprocessing import normalize_label  # type: ignore


CSV_COLUMNS = [
    "session_id",
    "label",
    "timestamp",
    "sensor",
    "room",
    "location",
    "status",
    "motion",
    "presence_score",
    "distance_mm",
]
SUBSCRIPTION_MESSAGE = json.dumps(
    {"type": "subscribe", "role": "dashboard"},
    ensure_ascii=False,
)


async def collect(args: argparse.Namespace) -> None:
    label = normalize_label(args.label)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0
    sample_interval = getattr(args, "sample_interval", 0.2)
    startup_timeout = getattr(args, "startup_timeout", 20.0)
    sample_timeout = getattr(args, "sample_timeout", 3.0)
    target_count = max(1, int(round(args.duration / sample_interval)))
    count = 0

    print(f"서버 연결 중: {args.server}")
    print(
        f"수집 시작: label={label}, session={args.session}, "
        f"목표={target_count}개({args.duration:.1f}초, {sample_interval:.1f}초 간격)"
    )
    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        async with websockets.connect(args.server) as websocket:
            await websocket.send(SUBSCRIPTION_MESSAGE)
            print("센서 데이터 수신 채널 구독 완료")
            loop = asyncio.get_running_loop()
            valid_sample_deadline = loop.time() + startup_timeout
            while count < target_count:
                remaining = valid_sample_deadline - loop.time()
                if remaining <= 0:
                    wait_description = "첫 측정값" if count == 0 else "다음 측정값"
                    raise TimeoutError(
                        f"{wait_description}을 제한 시간 내에 받지 못했습니다. "
                        "server.py와 serial_bridge.py 상태를 확인해주세요."
                    )
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError as exc:
                    wait_description = "첫 측정값" if count == 0 else "다음 측정값"
                    raise TimeoutError(
                        f"{wait_description}을 제한 시간 내에 받지 못했습니다. "
                        "server.py와 serial_bridge.py 상태를 확인해주세요."
                    ) from exc

                try:
                    data = json.loads(raw_message)
                except (json.JSONDecodeError, TypeError):
                    continue
                if data.get("type") != "radar_data":
                    continue
                if not args.include_inactive and data.get("status") is not True:
                    continue
                if args.room is not None and int(data.get("room", -1)) != args.room:
                    continue
                if args.location and data.get("location") != args.location:
                    continue

                writer.writerow(
                    {
                        "session_id": args.session,
                        "label": label,
                        "timestamp": data.get("timestamp")
                        or datetime.now(timezone.utc).isoformat(),
                        "sensor": data.get("sensor"),
                        "room": data.get("room"),
                        "location": data.get("location"),
                        "status": data.get("status"),
                        "motion": data.get("motion"),
                        "presence_score": data.get("presence_score"),
                        "distance_mm": data.get("distance_mm"),
                    }
                )
                stream.flush()
                count += 1
                valid_sample_deadline = loop.time() + sample_timeout
                print(
                    f"\r수집 진행: {count}/{target_count}개",
                    end="",
                    flush=True,
                )

    print(f"\n수집 완료: {destination} ({count}개)")
    if count == 0:
        print("경고: 조건에 맞는 radar_data를 받지 못했습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LifeSignal 학습용 센서 데이터 수집"
    )
    parser.add_argument("--server", default="ws://127.0.0.1:8881")
    parser.add_argument(
        "--label",
        required=True,
        help="human(사람), empty(빈 공간), dog(개)",
    )
    parser.add_argument("--session", required=True, help="독립 실험 세션 이름")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--sample-timeout", type=float, default=3.0)
    parser.add_argument("--output", default="AI/data/vpr100_samples.csv")
    parser.add_argument("--room", type=int)
    parser.add_argument("--location")
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="감지 해제 상태도 학습 CSV에 포함",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.duration <= 0
        or args.sample_interval <= 0
        or args.startup_timeout <= 0
        or args.sample_timeout <= 0
    ):
        raise SystemExit(
            "수집 시간, 수집 간격, 시작 대기 시간, "
            "측정값 대기 시간은 모두 0보다 커야 합니다."
        )
    try:
        asyncio.run(collect(args))
    except TimeoutError as exc:
        raise SystemExit(f"데이터 수집 실패: {exc}") from exc


if __name__ == "__main__":
    main()
