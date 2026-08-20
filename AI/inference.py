from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import joblib
import numpy as np

from AI.preprocessing import engineer_features, scale_sequences
from AI.priority import build_rescue_priority


TARGET_NAMES = {
    "human": "사람",
    "empty": "감지 대상 없음",
    "dog": "개",
    # 기존 2개 라벨 모델 호환용
    "pet": "반려동물",
}


@dataclass(frozen=True)
class ModelPrediction:
    target: str
    confidence: float


class SVMClassifier:
    def __init__(self, path: str | Path) -> None:
        artifact = joblib.load(path)
        if artifact.get("model_type") != "svm":
            raise ValueError("SVM 모델 파일이 아닙니다.")
        self.model = artifact["model"]
        self.channels = tuple(artifact["channels"])
        self.window_size = int(artifact["window_size"])
        self.model_name = "svm"
        self.sensor_type = artifact.get("sensor_type")
        self.trained_at = artifact.get("trained_at")

    def predict(self, window: np.ndarray) -> ModelPrediction:
        features, _ = engineer_features(window[None, :, :], self.channels)
        probabilities = self.model.predict_proba(features)[0]
        classes = self.model.classes_
        best_index = int(np.argmax(probabilities))
        return ModelPrediction(
            target=str(classes[best_index]),
            confidence=float(probabilities[best_index]),
        )


class CNNClassifier:
    def __init__(self, path: str | Path) -> None:
        import tensorflow as tf

        model_path = Path(path)
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"1D CNN 메타데이터를 찾을 수 없습니다: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model = tf.keras.models.load_model(model_path)
        self.channels = tuple(metadata["channels"])
        self.window_size = int(metadata["window_size"])
        self.mean = np.asarray(metadata["channel_mean"], dtype=np.float32)
        self.std = np.asarray(metadata["channel_std"], dtype=np.float32)
        self.labels = tuple(metadata.get("labels", ("pet", "human")))
        self.threshold = float(metadata.get("human_threshold", 0.5))
        self.model_name = "1d-cnn"
        self.sensor_type = metadata.get("sensor_type")
        self.trained_at = metadata.get("trained_at")

    def predict(self, window: np.ndarray) -> ModelPrediction:
        scaled = scale_sequences(window[None, :, :], self.mean, self.std)
        probabilities = np.asarray(
            self.model.predict(scaled, verbose=0),
            dtype=np.float32,
        )
        # 기존 sigmoid 모델도 계속 로드합니다.
        if probabilities.shape[-1] == 1:
            human_probability = float(probabilities.reshape(-1)[0])
            if human_probability >= self.threshold:
                return ModelPrediction("human", human_probability)
            return ModelPrediction("pet", 1.0 - human_probability)

        class_probabilities = probabilities[0]
        if len(class_probabilities) != len(self.labels):
            raise ValueError("1D CNN 라벨 수와 출력 크기가 다릅니다.")
        best_index = int(np.argmax(class_probabilities))
        return ModelPrediction(
            self.labels[best_index],
            float(class_probabilities[best_index]),
        )


def load_classifier(
    path: str | Path,
    model_type: str = "auto",
) -> SVMClassifier | CNNClassifier:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"AI 모델을 찾을 수 없습니다: {model_path}")
    selected_type = model_type
    if selected_type == "auto":
        selected_type = "cnn" if model_path.suffix == ".keras" else "svm"
    if selected_type == "svm":
        return SVMClassifier(model_path)
    if selected_type in {"cnn", "1d-cnn"}:
        return CNNClassifier(model_path)
    raise ValueError(f"지원하지 않는 모델 종류입니다: {model_type}")


class SensorAIEngine:
    def __init__(
        self,
        classifier: SVMClassifier | CNNClassifier | Any | None,
        *,
        update_interval: float = 5.0,
    ) -> None:
        if update_interval <= 0:
            raise ValueError("AI 갱신 주기는 0보다 커야 합니다.")
        self.classifier = classifier
        self.update_interval = update_interval
        self.channels = tuple(
            getattr(
                classifier,
                "channels",
                ("presence_score", "distance_mm", "motion", "status"),
            )
        )
        self.window_size = int(getattr(classifier, "window_size", 40))
        self.buffers: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.last_prediction_at: dict[str, float] = {}
        self.latest_results: dict[str, dict] = {}

    @staticmethod
    def sensor_key(data: dict) -> str:
        return (
            f"{data.get('sensor', 'unknown')}:"
            f"{data.get('room')}:{data.get('location')}"
        )

    @staticmethod
    def _number(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if np.isfinite(numeric) else 0.0

    def _sample(self, data: dict) -> np.ndarray:
        values: list[float] = []
        for channel in self.channels:
            value = data.get(channel)
            if channel in {"motion", "status"}:
                values.append(1.0 if value is True else 0.0)
            else:
                values.append(self._number(value))
        return np.asarray(values, dtype=np.float32)

    def enrich(self, data: dict, *, now: float | None = None) -> dict:
        if data.get("type") != "radar_data":
            return data
        now = monotonic() if now is None else now
        key = self.sensor_key(data)
        buffer = self.buffers[key]

        if self.classifier is None and data.get("status") is not True:
            buffer.clear()
            self.last_prediction_at.pop(key, None)
            self.latest_results.pop(key, None)
            enriched = dict(data)
            enriched["ai"] = self._pending_result("no_target", 0)
            enriched["rescue_priority"] = build_rescue_priority(enriched["ai"])
            return enriched

        buffer.append(self._sample(data))

        if self.classifier is None:
            ai_result = self._pending_result(
                "model_not_loaded",
                len(buffer),
            )
        elif len(buffer) < self.window_size:
            ai_result = self._pending_result(
                "collecting_window",
                len(buffer),
            )
        else:
            last_at = self.last_prediction_at.get(key)
            if last_at is None or now - last_at >= self.update_interval:
                prediction = self.classifier.predict(
                    np.asarray(buffer, dtype=np.float32)
                )
                ai_result = {
                    "ready": True,
                    "model": self.classifier.model_name,
                    "target": prediction.target,
                    "target_ko": TARGET_NAMES.get(
                        prediction.target,
                        prediction.target,
                    ),
                    "confidence": round(prediction.confidence, 4),
                    "window_samples": self.window_size,
                    "update_interval_sec": self.update_interval,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.latest_results[key] = ai_result
                self.last_prediction_at[key] = now
            else:
                ai_result = self.latest_results[key]

        enriched = dict(data)
        enriched["ai"] = dict(ai_result)
        enriched["rescue_priority"] = build_rescue_priority(
            enriched["ai"],
            human_risk=data.get("human_risk"),
        )
        return enriched

    def _pending_result(self, reason: str, sample_count: int) -> dict:
        return {
            "ready": False,
            "model": getattr(self.classifier, "model_name", None),
            "target": None,
            "target_ko": None,
            "confidence": None,
            "reason": reason,
            "samples_collected": sample_count,
            "samples_required": self.window_size,
            "update_interval_sec": self.update_interval,
        }
