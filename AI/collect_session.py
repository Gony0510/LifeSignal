"""프리셋 기반 V-PR100/C4001 학습 데이터 수집 도구."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from AI import collect_c4001_serial, collect_data
    from AI.collection_plan import (
        PROJECT_ROOT,
        CollectionPreset,
        get_preset,
        preset_names,
    )
except ModuleNotFoundError:
    import collect_c4001_serial  # type: ignore
    import collect_data  # type: ignore
    from collection_plan import (  # type: ignore
        PROJECT_ROOT,
        CollectionPreset,
        get_preset,
        preset_names,
    )


MANIFEST_PATH = PROJECT_ROOT / "AI/data/collection_manifest.json"


def _read_session_rows(path: Path, session_id: str) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if row.get("session_id") == session_id
        ]


def _read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _save_manifest(entry: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = _read_manifest()
    entries.append(entry)
    MANIFEST_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _display_preset(preset: CollectionPreset) -> None:
    print(f"수집 프리셋: {preset.name}")
    print(f"센서: {preset.sensor.upper()}")
    print(f"세션: {preset.session_id}")
    print(f"학습 라벨: {preset.label}")
    print(f"시연 시나리오: {preset.scenario}")
    print(f"CSV: {preset.output_path}")
    if preset.expected_room is not None:
        print(
            f"예상 위치: {preset.expected_room}호 "
            f"{preset.expected_location or '위치 미지정'}"
        )
    print(f"안내: {preset.instruction}")


def _build_args(preset: CollectionPreset, args: argparse.Namespace) -> argparse.Namespace:
    duration = preset.duration if args.duration is None else args.duration
    output = str(Path(args.output).expanduser()) if args.output else preset.output
    if not Path(output).is_absolute():
        output = str(PROJECT_ROOT / output)

    if preset.sensor == "vpr100":
        return argparse.Namespace(
            server=args.server,
            label=preset.label,
            session=preset.session_id,
            duration=duration,
            output=output,
            room=args.room if args.room is not None else preset.expected_room,
            location=(
                args.location
                if args.location is not None
                else preset.expected_location
            ),
            include_inactive=preset.include_inactive,
        )

    return argparse.Namespace(
        port=args.port,
        baud=args.baud,
        label=preset.label,
        session=preset.session_id,
        duration=duration,
        startup_timeout=args.startup_timeout,
        sample_timeout=args.sample_timeout,
        reconnect_interval=args.reconnect_interval,
        boot_wait=args.boot_wait,
        output=output,
    )


def _validate_args(preset: CollectionPreset, args: argparse.Namespace) -> None:
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("수집 시간은 0보다 커야 합니다.")
    if preset.sensor == "c4001" and args.server != "ws://127.0.0.1:8881":
        raise SystemExit("C4001 수집은 USB 시리얼을 사용하므로 --server를 사용하지 않습니다.")
    if preset.sensor == "vpr100" and args.port:
        raise SystemExit("V-PR100 프리셋은 서버를 통해 수집하므로 --port를 사용하지 않습니다.")


def _run_collection(preset: CollectionPreset, args: argparse.Namespace) -> int:
    destination = (
        Path(args.output).expanduser()
        if args.output
        else preset.output_path
    )
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    _display_preset(preset)
    if args.dry_run:
        print("\n[dry-run] 실제 수집은 실행하지 않았습니다.")
        return 0

    existing = _read_session_rows(destination, preset.session_id)
    if existing and not args.allow_existing_session:
        raise SystemExit(
            f"세션 {preset.session_id!r}이 이미 {destination}에 "
            f"{len(existing)}행 있습니다. 새 세션 이름을 사용하거나 "
            "정말 이어서 저장하려면 --allow-existing-session을 사용하세요."
        )

    collector_args = _build_args(preset, args)
    if preset.sensor == "vpr100":
        asyncio.run(collect_data.collect(collector_args))
    else:
        collect_c4001_serial.collect(collector_args)

    rows = _read_session_rows(destination, preset.session_id)
    if not rows:
        raise SystemExit(
            f"세션 {preset.session_id!r}의 수집 행이 없습니다. "
            "센서 연결과 라벨 대상을 확인해주세요."
        )

    _save_manifest(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "preset": preset.name,
            "sensor": preset.sensor,
            "session_id": preset.session_id,
            "label": preset.label,
            "scenario": preset.scenario,
            "rows": len(rows),
            "output": str(destination),
            "expected_room": preset.expected_room,
            "expected_location": preset.expected_location,
        }
    )
    print(f"수집 기록 저장: {MANIFEST_PATH}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="시연 배치용 V-PR100/C4001 데이터 수집 프리셋 실행"
    )
    parser.add_argument("--preset", choices=preset_names())
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 수집 프리셋을 표시하고 종료",
    )
    parser.add_argument("--duration", type=float, help="기본 60초를 덮어쓸 수집 시간")
    parser.add_argument("--output", help="프리셋 기본 CSV 경로를 덮어쓸 경로")
    parser.add_argument("--server", default="ws://127.0.0.1:8881")
    parser.add_argument("--port", help="C4001 USB 시리얼 포트")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--room", type=int, help="V-PR100 서버 데이터 필터 호수")
    parser.add_argument("--location", help="V-PR100 서버 데이터 필터 위치")
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--sample-timeout", type=float, default=3.0)
    parser.add_argument("--reconnect-interval", type=float, default=2.0)
    parser.add_argument("--boot-wait", type=float, default=2.0)
    parser.add_argument(
        "--allow-existing-session",
        action="store_true",
        help="이미 존재하는 세션에 이어서 저장",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="수집 대상과 설정만 확인하고 실제 센서에는 연결하지 않음",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name in preset_names():
            preset = get_preset(name)
            print(
                f"{name}: label={preset.label}, "
                f"session={preset.session_id}, "
                f"scenario={preset.scenario}, output={preset.output}"
            )
        return
    if not args.preset:
        raise SystemExit("--preset을 지정하거나 --list로 수집 목록을 확인해주세요.")

    preset = get_preset(args.preset)
    _validate_args(preset, args)
    _run_collection(preset, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n데이터 수집을 중단했습니다.")
