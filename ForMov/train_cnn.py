from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    from AI.augmentation import augment_training_windows
    from ForMov.config import (
        CHANNELS,
        DEFAULT_CNN_PATH,
        DEFAULT_DATA_PATH,
        STEP_SIZE,
        SUPPORTED_LABELS,
        WINDOW_SIZE,
    )
    from ForMov.preprocessing import (
        build_windows,
        fit_channel_scaler,
        grouped_train_test_indices,
        load_sensor_csv,
        scale_sequences,
        validate_training_data,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from AI.augmentation import augment_training_windows
    from ForMov.config import (  # type: ignore
        CHANNELS,
        DEFAULT_CNN_PATH,
        DEFAULT_DATA_PATH,
        STEP_SIZE,
        SUPPORTED_LABELS,
        WINDOW_SIZE,
    )
    from ForMov.preprocessing import (  # type: ignore
        build_windows,
        fit_channel_scaler,
        grouped_train_test_indices,
        load_sensor_csv,
        scale_sequences,
        validate_training_data,
    )


def build_model(
    window_size: int,
    channel_count: int,
    class_count: int,
    *,
    model_name: str = "lifesignal_formov_c4001_cnn",
):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(window_size, channel_count))
    x = tf.keras.layers.Conv1D(32, 5, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(
    data_paths: list[str],
    output_path: str,
    *,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    test_size: float = 0.25,
    epochs: int = 80,
    batch_size: int = 32,
    random_state: int = 42,
    augment_copies: int = 0,
    augment_jitter: float = 0.03,
    augment_scale: float = 0.05,
    channels: tuple[str, ...] = CHANNELS,
    sensor_type: str = "c4001",
    task: str = "c4001_rescue_priority_scenario",
    display_name: str = "C4001",
):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "1D CNN 학습에는 tensorflow가 필요합니다. "
            "SVM만 사용할 경우에는 설치하지 않아도 됩니다."
        ) from exc

    np.random.seed(random_state)
    tf.random.set_seed(random_state)
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
    original_train = dataset.samples[train_idx]
    training_samples, training_labels = augment_training_windows(
        original_train,
        dataset.labels[train_idx],
        dataset.channels,
        copies=augment_copies,
        jitter=augment_jitter,
        scale=augment_scale,
        random_state=random_state,
    )
    mean, std = fit_channel_scaler(original_train)
    x_train = scale_sequences(training_samples, mean, std)
    x_test = scale_sequences(dataset.samples[test_idx], mean, std)
    label_to_index = {label: index for index, label in enumerate(SUPPORTED_LABELS)}
    y_train = np.asarray([label_to_index[label] for label in training_labels])
    y_test = np.asarray([label_to_index[label] for label in dataset.labels[test_idx]])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(SUPPORTED_LABELS)),
        y=y_train,
    )
    class_weight = {index: float(weight) for index, weight in enumerate(weights)}

    model = build_model(
        window_size,
        len(channels),
        len(SUPPORTED_LABELS),
        model_name=f"lifesignal_formov_{sensor_type}_cnn",
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
            ),
        ],
        verbose=2,
    )
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.asarray(SUPPORTED_LABELS)[np.argmax(probabilities, axis=1)]
    print(f"\n[영상용 {display_name} 1D CNN 검증 결과]")
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
            dataset.labels[test_idx], predictions, labels=list(SUPPORTED_LABELS)
        )
    )

    destination = Path(output_path)
    if destination.suffix != ".keras":
        destination = destination.with_suffix(".keras")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    metadata = {
        "artifact_version": 1,
        "model_type": "cnn",
        "task": task,
        "sensor_type": sensor_type,
        "channels": list(dataset.channels),
        "window_size": dataset.window_size,
        "step_size": step_size,
        "labels": list(SUPPORTED_LABELS),
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
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
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"1D CNN 모델 저장 완료: {destination}")
    print(f"1D CNN 메타데이터 저장 완료: {metadata_path}")
    return model, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상용 C4001 수면/대피/쓰러짐/무신호 1D CNN 학습"
    )
    parser.add_argument("--data", nargs="+", default=[str(DEFAULT_DATA_PATH)])
    parser.add_argument("--output", default=str(DEFAULT_CNN_PATH))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=STEP_SIZE)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--augment-copies", type=int, default=0)
    parser.add_argument("--augment-jitter", type=float, default=0.03)
    parser.add_argument("--augment-scale", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_cnn(
            args.data,
            args.output,
            window_size=args.window_size,
            step_size=args.step_size,
            test_size=args.test_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.random_state,
            augment_copies=args.augment_copies,
            augment_jitter=args.augment_jitter,
            augment_scale=args.augment_scale,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"영상용 1D CNN 학습 실패: {exc}") from exc


if __name__ == "__main__":
    main()
