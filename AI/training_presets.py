from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from AI.preprocessing import (
        SUPPORTED_LABELS,
        load_sensor_csv,
        normalize_label,
        training_group_ids,
    )
    from AI.svm import train_svm
except ModuleNotFoundError:
    from preprocessing import (  # type: ignore
        SUPPORTED_LABELS,
        load_sensor_csv,
        normalize_label,
        training_group_ids,
    )
    from svm import train_svm  # type: ignore


@dataclass(frozen=True)
class SensorTrainingPreset:
    sensor_type: str
    display_name: str
    channels: tuple[str, ...]
    required_metadata: tuple[str, ...]
    value_channel: str
    default_data: str
    default_svm_output: str
    default_cnn_output: str
    window_size: int
    step_size: int


VPR100_PRESET = SensorTrainingPreset(
    sensor_type="vpr100",
    display_name="V-PR100",
    channels=("presence_score", "status"),
    required_metadata=("session_id", "label", "sensor"),
    value_channel="presence_score",
    default_data="AI/data/vpr100_samples.csv",
    default_svm_output="AI/artifacts/vpr100-svm.joblib",
    default_cnn_output="AI/artifacts/vpr100-cnn.keras",
    window_size=25,
    step_size=5,
)

C4001_PRESET = SensorTrainingPreset(
    sensor_type="c4001",
    display_name="C4001",
    channels=("target_energy", "status"),
    required_metadata=("session_id", "label", "sensor_id"),
    value_channel="target_energy",
    default_data="AI/data/c4001_samples.csv",
    default_svm_output="AI/artifacts/c4001-svm.joblib",
    default_cnn_output="AI/artifacts/c4001-cnn.keras",
    window_size=25,
    step_size=5,
)


def validate_training_data(
    data_paths: list[str],
    preset: SensorTrainingPreset,
) -> dict[str, object]:
    frame = load_sensor_csv(data_paths)
    required = {
        *preset.required_metadata,
        *preset.channels,
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{preset.display_name} 학습 CSV에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    normalized_labels = frame["label"].map(normalize_label)
    available_labels = set(normalized_labels.unique())
    missing_labels = [
        label for label in SUPPORTED_LABELS if label not in available_labels
    ]
    if missing_labels:
        raise ValueError(
            "학습 불가: "
            + ", ".join(missing_labels)
            + " 데이터가 없습니다."
        )

    session_labels = pd.DataFrame(
        {
            "session_id": frame["session_id"].astype(str),
            "training_group_id": training_group_ids(frame),
            "label": normalized_labels,
        }
    )
    labels_per_session = session_labels.groupby("session_id")["label"].nunique()
    mixed_sessions = labels_per_session[labels_per_session > 1].index.tolist()
    if mixed_sessions:
        raise ValueError(
            "한 세션에 여러 라벨이 포함되어 있습니다: "
            + ", ".join(mixed_sessions)
        )

    labels_per_group = session_labels.groupby("training_group_id")["label"].nunique()
    mixed_groups = labels_per_group[labels_per_group > 1].index.tolist()
    if mixed_groups:
        raise ValueError(
            "같은 원본 세션에서 파생된 데이터에 여러 라벨이 있습니다: "
            + ", ".join(mixed_groups)
        )

    session_counts = (
        session_labels[["training_group_id", "label"]].drop_duplicates()
        .groupby("label")["training_group_id"]
        .nunique()
        .to_dict()
    )
    insufficient = [
        label
        for label in SUPPORTED_LABELS
        if int(session_counts.get(label, 0)) < 2
    ]
    if insufficient:
        details = ", ".join(
            f"{label}={int(session_counts.get(label, 0))}세션"
            for label in insufficient
        )
        raise ValueError(
            "학습/검증 분리를 위해 각 라벨이 최소 2세션 필요합니다: "
            + details
        )

    numeric_values = pd.to_numeric(frame[preset.value_channel], errors="coerce")
    invalid_count = int(numeric_values.isna().sum())
    if invalid_count:
        raise ValueError(
            f"{preset.value_channel}에 숫자가 아닌 값이 "
            f"{invalid_count}개 있습니다."
        )

    summary = {
        "sensor_type": preset.sensor_type,
        "rows": int(len(frame)),
        "sessions": int(session_labels["training_group_id"].nunique()),
        "actual_session_ids": int(frame["session_id"].astype(str).nunique()),
        "session_counts": {
            label: int(session_counts.get(label, 0))
            for label in SUPPORTED_LABELS
        },
        "channels": preset.channels,
    }
    print(
        f"{preset.display_name} 데이터 검증 완료: "
        f"{summary['rows']}행, {summary['sessions']}개 독립 세션, "
        f"채널={', '.join(preset.channels)}"
    )
    return summary


def _effective_test_size(
    requested_test_size: float,
    summary: dict[str, object],
) -> float:
    session_count = int(summary["sessions"])
    minimum_fraction = len(SUPPORTED_LABELS) / session_count
    effective = max(requested_test_size, minimum_fraction)
    if effective > requested_test_size:
        print(
            "세 라벨을 검증 세트에 모두 포함하기 위해 "
            f"검증 비율을 {requested_test_size:.2f}에서 "
            f"{effective:.2f}(으)로 자동 조정합니다."
        )
    return effective


def _common_parser(preset: SensorTrainingPreset, model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"LifeSignal {preset.display_name} 전용 "
            f"empty/human/dog {model_name} 학습"
        )
    )
    parser.add_argument("--data", nargs="+", default=[preset.default_data])
    parser.add_argument(
        "--window-size",
        type=int,
        default=preset.window_size,
        help="한 학습 윈도우에 포함할 센서 샘플 수",
    )
    parser.add_argument("--step-size", type=int, default=preset.step_size)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=0,
        help="학습 윈도우당 생성할 변형 사본 수(기본 0: 사용 안 함)",
    )
    parser.add_argument("--augment-jitter", type=float, default=0.03)
    parser.add_argument("--augment-scale", type=float, default=0.05)
    return parser


def run_svm_cli(preset: SensorTrainingPreset) -> None:
    parser = _common_parser(preset, "SVM")
    parser.add_argument("--output", default=preset.default_svm_output)
    args = parser.parse_args()
    try:
        summary = validate_training_data(args.data, preset)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(f"{preset.display_name} 학습 준비 실패: {exc}")
    test_size = _effective_test_size(args.test_size, summary)
    train_svm(
        args.data,
        args.output,
        window_size=args.window_size,
        step_size=args.step_size,
        channels=preset.channels,
        test_size=test_size,
        random_state=args.random_state,
        sensor_type=preset.sensor_type,
        augment_copies=args.augment_copies,
        augment_jitter=args.augment_jitter,
        augment_scale=args.augment_scale,
    )


def _load_cnn_trainer() -> Callable[..., object]:
    module_path = Path(__file__).with_name("1d-cnn.py")
    spec = importlib.util.spec_from_file_location(
        "lifesignal_shared_1d_cnn",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"1D CNN 학습 모듈을 불러올 수 없습니다: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train_cnn


def run_cnn_cli(preset: SensorTrainingPreset) -> None:
    parser = _common_parser(preset, "1D CNN")
    parser.add_argument("--output", default=preset.default_cnn_output)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        summary = validate_training_data(args.data, preset)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(f"{preset.display_name} 학습 준비 실패: {exc}")
    test_size = _effective_test_size(args.test_size, summary)
    train_cnn = _load_cnn_trainer()
    train_cnn(
        args.data,
        args.output,
        window_size=args.window_size,
        step_size=args.step_size,
        channels=preset.channels,
        test_size=test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        sensor_type=preset.sensor_type,
        augment_copies=args.augment_copies,
        augment_jitter=args.augment_jitter,
        augment_scale=args.augment_scale,
    )
