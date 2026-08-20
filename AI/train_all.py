"""수집 완료된 센서 CSV로 기본 SVM 모델을 한 번에 학습합니다."""

from __future__ import annotations

import argparse
from dataclasses import replace

try:
    from AI.svm import train_svm
    from AI.training_presets import (
        C4001_PRESET,
        VPR100_PRESET,
        SensorTrainingPreset,
        _effective_test_size,
        validate_training_data,
    )
except ModuleNotFoundError:
    from svm import train_svm  # type: ignore
    from training_presets import (  # type: ignore
        C4001_PRESET,
        VPR100_PRESET,
        SensorTrainingPreset,
        _effective_test_size,
        validate_training_data,
    )


def train_preset(
    preset: SensorTrainingPreset,
    *,
    augment_copies: int = 0,
    random_state: int = 42,
) -> None:
    data_paths = [preset.default_data]
    summary = validate_training_data(data_paths, preset)
    test_size = _effective_test_size(0.25, summary)
    train_svm(
        data_paths,
        preset.default_svm_output,
        window_size=preset.window_size,
        step_size=preset.step_size,
        channels=preset.channels,
        test_size=test_size,
        random_state=random_state,
        sensor_type=preset.sensor_type,
        augment_copies=augment_copies,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LifeSignal V-PR100/C4001 기본 SVM 모델 학습",
    )
    parser.add_argument(
        "--sensor",
        choices=["all", "vpr100", "c4001"],
        default="all",
    )
    parser.add_argument("--vpr100-data", default=VPR100_PRESET.default_data)
    parser.add_argument("--c4001-data", default=C4001_PRESET.default_data)
    parser.add_argument("--augment-copies", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.augment_copies < 0:
        raise SystemExit("어그멘테이션 사본 수는 0 이상이어야 합니다.")

    presets = {
        "vpr100": replace(VPR100_PRESET, default_data=args.vpr100_data),
        "c4001": replace(C4001_PRESET, default_data=args.c4001_data),
    }
    selected = presets if args.sensor == "all" else {args.sensor: presets[args.sensor]}

    for family, preset in selected.items():
        print(f"\n[{preset.display_name} 모델 학습 시작]")
        try:
            train_preset(
                preset,
                augment_copies=args.augment_copies,
                random_state=args.random_state,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"{family} 모델 학습 준비 실패: {exc}") from exc
    print("\n선택한 센서 모델 학습이 모두 완료되었습니다.")


if __name__ == "__main__":
    main()
