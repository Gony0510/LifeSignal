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


async def collect(args: argparse.Namespace) -> None:
    label = normalize_label(args.label)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0
    deadline = asyncio.get_running_loop().time() + args.duration
    count = 0

    print(f"서버 연결 중: {args.server}")
    print(
        f"수집 시작: label={label}, session={args.session}, "
        f"duration={args.duration:.1f}초"
    )
    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        async with websockets.connect(args.server) as websocket:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    break

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
                print(f"\r수집된 측정값: {count}", end="", flush=True)

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
    if args.duration <= 0:
        raise SystemExit("수집 시간은 0보다 커야 합니다.")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
