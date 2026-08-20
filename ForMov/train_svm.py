from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from AI.augmentation import augment_training_windows
    from ForMov.config import (
        CHANNELS,
        DEFAULT_DATA_PATH,
        DEFAULT_SVM_PATH,
        STEP_SIZE,
        SUPPORTED_LABELS,
        WINDOW_SIZE,
    )
    from ForMov.preprocessing import (
        build_windows,
        engineer_features,
        grouped_train_test_indices,
        load_sensor_csv,
        validate_training_data,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from AI.augmentation import augment_training_windows
    from ForMov.config import (  # type: ignore
        CHANNELS,
        DEFAULT_DATA_PATH,
        DEFAULT_SVM_PATH,
        STEP_SIZE,
        SUPPORTED_LABELS,
        WINDOW_SIZE,
    )
from ForMov.preprocessing import (  # type: ignore
        build_windows,
        engineer_features,
        grouped_train_test_indices,
        load_sensor_csv,
        validate_training_data,
    )


def build_svm_model(random_state: int) -> Pipeline:
    return Pipeline(
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


def train_svm(
    data_paths: list[str],
    output_path: str,
    *,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    test_size: float = 0.25,
    random_state: int = 42,
    augment_copies: int = 0,
    augment_jitter: float = 0.03,
    augment_scale: float = 0.05,
    channels: tuple[str, ...] = CHANNELS,
    sensor_type: str = "c4001",
    task: str = "c4001_rescue_priority_scenario",
    display_name: str = "C4001",
) -> dict:
    frame = load_sensor_csv(data_paths)
    summary = validate_training_data(frame, channels)
    print(
        "영상용 데이터 검증 완료: "
        f"{summary['rows']:,}행, {summary['sessions']}개 독립 세션"
    )
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
    x_train, feature_names = engineer_features(training_samples, dataset.channels)
    x_test, _ = engineer_features(dataset.samples[test_idx], dataset.channels)

    validation_model = build_svm_model(random_state)
    validation_model.fit(x_train, training_labels)
    predictions = validation_model.predict(x_test)
    print(f"\n[영상용 {display_name} SVM 검증 결과]")
    print(
        classification_report(
            dataset.labels[test_idx],
            predictions,
            labels=list(SUPPORTED_LABELS),
            digits=4,
            zero_division=0,
        )
    )
    print("혼동행렬 (행=실제, 열=예측 / " + ", ".join(SUPPORTED_LABELS) + ")")
    print(
        confusion_matrix(
            dataset.labels[test_idx],
            predictions,
            labels=list(SUPPORTED_LABELS),
        )
    )

    final_samples, final_labels = augment_training_windows(
        dataset.samples,
        dataset.labels,
        dataset.channels,
        copies=augment_copies,
        jitter=augment_jitter,
        scale=augment_scale,
        random_state=random_state,
    )
    x_final, feature_names = engineer_features(final_samples, dataset.channels)
    model = build_svm_model(random_state)
    model.fit(x_final, final_labels)
    print(
        "최종 모델 전체 데이터 재학습 완료: "
        f"원본 윈도우 {len(dataset.samples):,}개, "
        f"최종 학습 윈도우 {len(final_samples):,}개"
    )

    artifact = {
        "artifact_version": 1,
        "model_type": "svm",
        "task": task,
        "sensor_type": sensor_type,
        "model": model,
        "channels": list(dataset.channels),
        "window_size": dataset.window_size,
        "step_size": step_size,
        "feature_names": feature_names,
        "labels": list(model.named_steps["classifier"].classes_),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_files": data_paths,
        "training_windows": int(len(final_samples)),
        "original_training_windows": int(len(dataset.samples)),
        "validation_training_windows": int(len(training_samples)),
        "validation_original_training_windows": int(len(train_idx)),
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
        description="영상용 C4001 수면/대피/쓰러짐/무신호 SVM 학습"
    )
    parser.add_argument("--data", nargs="+", default=[str(DEFAULT_DATA_PATH)])
    parser.add_argument("--output", default=str(DEFAULT_SVM_PATH))
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
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"영상용 SVM 학습 실패: {exc}") from exc


if __name__ == "__main__":
    main()
