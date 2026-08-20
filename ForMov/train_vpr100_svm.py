from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ForMov.config import (  # noqa: E402
    STEP_SIZE,
    VPR100_CHANNELS,
    VPR100_DEFAULT_DATA_PATH,
    VPR100_DEFAULT_SVM_PATH,
    WINDOW_SIZE,
)
from ForMov.train_svm import train_svm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상용 V-PR100 수면/대피/쓰러짐/무신호 SVM 학습"
    )
    parser.add_argument("--data", nargs="+", default=[str(VPR100_DEFAULT_DATA_PATH)])
    parser.add_argument("--output", default=str(VPR100_DEFAULT_SVM_PATH))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=STEP_SIZE)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--augment-copies", type=int, default=0)
    parser.add_argument("--augment-jitter", type=float, default=0.03)
    parser.add_argument("--augment-scale", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_svm(
            args.data,
            args.output,
            window_size=args.window_size,
            step_size=args.step_size,
            test_size=args.test_size,
            random_state=args.random_state,
            augment_copies=args.augment_copies,
            augment_jitter=args.augment_jitter,
            augment_scale=args.augment_scale,
            channels=VPR100_CHANNELS,
            sensor_type="vpr100",
            task="vpr100_rescue_priority_scenario",
            display_name="V-PR100",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"영상용 V-PR100 SVM 학습 실패: {exc}") from exc


if __name__ == "__main__":
    main()
