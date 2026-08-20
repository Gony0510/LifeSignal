from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from AI.augment_sensor_csv import augment_csv_file  # noqa: E402
from AI.training_presets import SensorTrainingPreset  # noqa: E402
from ForMov.config import (  # noqa: E402
    STEP_SIZE,
    VPR100_CHANNELS,
    VPR100_DEFAULT_CNN_PATH,
    VPR100_DEFAULT_DATA_PATH,
    VPR100_DEFAULT_SVM_PATH,
    WINDOW_SIZE,
)


VPR100_FORMOV_PRESET = SensorTrainingPreset(
    sensor_type="vpr100",
    display_name="영상용 V-PR100",
    channels=VPR100_CHANNELS,
    required_metadata=("session_id", "label", "sensor"),
    value_channel="presence_score",
    default_data=str(VPR100_DEFAULT_DATA_PATH),
    default_svm_output=str(VPR100_DEFAULT_SVM_PATH),
    default_cnn_output=str(VPR100_DEFAULT_CNN_PATH),
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "영상용 V-PR100 CSV에 원하는 개수만큼 증강 데이터를 추가합니다. "
            "--apply가 없으면 미리보기만 합니다."
        )
    )
    parser.add_argument("--data", default=str(VPR100_DEFAULT_DATA_PATH))
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = augment_csv_file(
            args.data,
            VPR100_FORMOV_PRESET,
            sessions=args.sessions,
            count=args.count,
            jitter=args.jitter,
            scale=args.scale,
            random_state=args.random_state,
            apply=args.apply,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"영상용 V-PR100 데이터 어그멘테이션 실패: {exc}") from exc

    mode = "적용 완료" if result.applied else "미리보기"
    print(f"\n[{mode}] 영상용 V-PR100 CSV 어그멘테이션")
    print(f"대상 파일: {result.data_path}")
    print(f"원본 세션: {', '.join(result.source_sessions)}")
    print(f"추가 행: {result.generated_rows:,}개")
    print(f"완료 후 전체 행: {result.total_rows:,}개")
    if result.backup_path:
        print(f"원본 백업: {result.backup_path}")
    if not result.applied:
        print("CSV는 변경하지 않았습니다. 적용하려면 --apply를 추가하세요.")


if __name__ == "__main__":
    main()
