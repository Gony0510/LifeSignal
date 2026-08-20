from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from AI.augmentation import augment_training_windows
    from AI.preprocessing import (
        DEFAULT_CHANNELS,
        SUPPORTED_LABELS,
        build_windows,
        engineer_features,
        grouped_train_test_indices,
        load_sensor_csv,
    )
except ModuleNotFoundError:
    from augmentation import augment_training_windows  # type: ignore
    from preprocessing import (  # type: ignore
        DEFAULT_CHANNELS,
        SUPPORTED_LABELS,
        build_windows,
        engineer_features,
        grouped_train_test_indices,
        load_sensor_csv,
    )


def train_svm(
    data_paths: list[str],
    output_path: str,
    *,
    window_size: int = 40,
    step_size: int = 10,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    test_size: float = 0.25,
    random_state: int = 42,
    sensor_type: str | None = None,
    augment_copies: int = 0,
    augment_jitter: float = 0.03,
    augment_scale: float = 0.05,
) -> dict:
    frame = load_sensor_csv(data_paths)
    dataset = build_windows(
        frame,
        window_size=window_size,
        step_size=step_size,
        channels=channels,
    )
    train_idx, test_idx = grouped_train_test_indices(
        dataset.labels,
        dataset.groups,
        test_size=test_size,
        random_state=random_state,
    )
    training_samples, training_labels = augment_training_windows(
        dataset.samples[train_idx],
        dataset.labels[train_idx],
        dataset.channels,
        copies=augment_copies,
        jitter=augment_jitter,
        scale=augment_scale,
        random_state=random_state,
    )
    training_features, feature_names = engineer_features(
        training_samples,
        dataset.channels,
    )
    validation_features, _ = engineer_features(
        dataset.samples[test_idx],
        dataset.channels,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    model.fit(training_features, training_labels)
    predictions = model.predict(validation_features)

    print("\n[SVM 검증 결과]")
    print(classification_report(dataset.labels[test_idx], predictions, digits=4))
    print(
        "혼동행렬 (행=실제, 열=예측 / "
        + ", ".join(SUPPORTED_LABELS)
        + ")"
    )
    print(
        confusion_matrix(
            dataset.labels[test_idx],
            predictions,
            labels=list(SUPPORTED_LABELS),
        )
    )

    artifact = {
        "artifact_version": 1,
        "model_type": "svm",
        "sensor_type": sensor_type,
        "model": model,
        "channels": list(dataset.channels),
        "window_size": dataset.window_size,
        "step_size": step_size,
        "feature_names": feature_names,
        "labels": list(model.named_steps["classifier"].classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_files": data_paths,
        "training_windows": int(len(training_samples)),
        "original_training_windows": int(len(train_idx)),
        "validation_windows": int(len(test_idx)),
        "augmentation": {
            "copies": augment_copies,
            "jitter": augment_jitter,
            "scale": augment_scale,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    print(f"SVM 모델 저장 완료: {destination}")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LifeSignal 사람/빈 공간/개 SVM 학습"
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="수집한 센서 CSV 파일 경로",
    )
    parser.add_argument(
        "--output",
        default="AI/artifacts/svm.joblib",
        help="저장할 모델 경로",
    )
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--augment-copies", type=int, default=0)
    parser.add_argument("--augment-jitter", type=float, default=0.03)
    parser.add_argument("--augment-scale", type=float, default=0.05)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(DEFAULT_CHANNELS),
        help="학습에 사용할 시계열 채널",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_svm(
        args.data,
        args.output,
        window_size=args.window_size,
        step_size=args.step_size,
        channels=tuple(args.channels),
        test_size=args.test_size,
        random_state=args.random_state,
        augment_copies=args.augment_copies,
        augment_jitter=args.augment_jitter,
        augment_scale=args.augment_scale,
    )


if __name__ == "__main__":
    main()
