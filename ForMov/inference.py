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

try:
    from ForMov.config import CHANNELS, LABEL_NAMES_KO, STEP_SIZE, WINDOW_SIZE
    from ForMov.preprocessing import engineer_features, scale_sequences
    from ForMov.priority import RescuePriorityEngine
except ModuleNotFoundError:
    from config import CHANNELS, LABEL_NAMES_KO, STEP_SIZE, WINDOW_SIZE  # type: ignore
    from preprocessing import engineer_features, scale_sequences  # type: ignore
    from priority import RescuePriorityEngine  # type: ignore


@dataclass(frozen=True)
class ModelPrediction:
    scenario: str
    confidence: float


@dataclass(frozen=True)
class TimedSample:
    values: np.ndarray
    received_at: float


class SVMClassifier:
    def __init__(self, path: str | Path) -> None:
        artifact = joblib.load(path)
        if artifact.get("model_type") != "svm":
            raise ValueError("영상용 SVM 모델 파일이 아닙니다.")
        if artifact.get("task") not in {
            "c4001_rescue_priority_scenario",
            "vpr100_rescue_priority_scenario",
        }:
            raise ValueError("영상용 구조 우선순위 모델이 아닙니다.")
        self.sensor_type = str(artifact.get("sensor_type", "c4001"))
        self.model = artifact["model"]
        self.channels = tuple(artifact["channels"])
        self.window_size = int(artifact["window_size"])
        self.step_size = int(artifact.get("step_size", STEP_SIZE))
        self.model_name = "svm"

    def predict(self, window: np.ndarray) -> ModelPrediction:
        features, _ = engineer_features(window[None, :, :], self.channels)
        probabilities = self.model.predict_proba(features)[0]
        classes = self.model.classes_
        best = int(np.argmax(probabilities))
        return ModelPrediction(str(classes[best]), float(probabilities[best]))


class CNNClassifier:
    def __init__(self, path: str | Path) -> None:
        import tensorflow as tf

        model_path = Path(path)
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"CNN 메타데이터를 찾을 수 없습니다: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("task") not in {
            "c4001_rescue_priority_scenario",
            "vpr100_rescue_priority_scenario",
        }:
            raise ValueError("영상용 구조 우선순위 CNN이 아닙니다.")
        self.sensor_type = str(metadata.get("sensor_type", "c4001"))
        self.model = tf.keras.models.load_model(model_path)
        self.channels = tuple(metadata["channels"])
        self.window_size = int(metadata["window_size"])
        self.step_size = int(metadata.get("step_size", STEP_SIZE))
        self.mean = np.asarray(metadata["channel_mean"], dtype=np.float32)
        self.std = np.asarray(metadata["channel_std"], dtype=np.float32)
        self.labels = tuple(metadata["labels"])
        self.model_name = "1d-cnn"

    def predict(self, window: np.ndarray) -> ModelPrediction:
        scaled = scale_sequences(window[None, :, :], self.mean, self.std)
        probabilities = np.asarray(self.model.predict(scaled, verbose=0))[0]
        best = int(np.argmax(probabilities))
        return ModelPrediction(self.labels[best], float(probabilities[best]))


def load_classifier(path: str | Path, model_type: str = "auto") -> SVMClassifier | CNNClassifier:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"AI 모델을 찾을 수 없습니다: {model_path}")
    selected = model_type
    if selected == "auto":
        selected = "cnn" if model_path.suffix == ".keras" else "svm"
    if selected == "svm":
        return SVMClassifier(model_path)
    if selected in {"cnn", "1d-cnn"}:
        return CNNClassifier(model_path)
    raise ValueError(f"지원하지 않는 모델 종류입니다: {model_type}")


class SensorAIEngine:
    def __init__(
        self,
        classifier: SVMClassifier | CNNClassifier | Any | None,
        *,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        priority_engine: RescuePriorityEngine | None = None,
    ) -> None:
        if step_size < 1:
            raise ValueError("step_size는 1 이상이어야 합니다.")
        self.classifier = classifier
        self.channels = tuple(getattr(classifier, "channels", CHANNELS))
        self.window_size = int(getattr(classifier, "window_size", window_size))
        self.step_size = int(getattr(classifier, "step_size", step_size))
        self.buffers: dict[str, deque[TimedSample]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.samples_since_prediction: dict[str, int] = defaultdict(int)
        self.latest_results: dict[str, dict[str, object]] = {}
        self.last_accepted_predictions: dict[str, ModelPrediction] = {}
        self.priority_engine = priority_engine or RescuePriorityEngine()

    def _accept_prediction(
        self,
        key: str,
        prediction: ModelPrediction,
    ) -> tuple[ModelPrediction | None, bool]:
        """Accept fallen only after an evacuating -> fallen transition.

        Once a fallen episode has been confirmed, consecutive fallen predictions
        remain valid until another scenario is accepted. A blocked fallen result
        keeps the last accepted non-fallen scenario without relabelling the raw
        prediction.
        """
        previous = self.last_accepted_predictions.get(key)
        if prediction.scenario != "fallen":
            self.last_accepted_predictions[key] = prediction
            return prediction, False
        if previous is not None and previous.scenario in {"evacuating", "fallen"}:
            self.last_accepted_predictions[key] = prediction
            return prediction, False
        return previous, True

    @staticmethod
    def sensor_key(data: dict[str, object]) -> str:
        return f"{data.get('sensor', 'C4001')}:{data.get('room')}:{data.get('location')}"

    @staticmethod
    def _number(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if np.isfinite(numeric) else 0.0

    def _sample(self, data: dict[str, object]) -> np.ndarray:
        values: list[float] = []
        for channel in self.channels:
            value = data.get(channel)
            if channel == "status":
                values.append(1.0 if value is True else 0.0)
            else:
                values.append(self._number(value))
        return np.asarray(values, dtype=np.float32)

    def enrich(self, data: dict[str, object], *, now: float | None = None) -> dict[str, object]:
        if data.get("type") != "radar_data":
            return data
        current = monotonic() if now is None else now
        key = self.sensor_key(data)
        buffer = self.buffers[key]
        buffer.append(TimedSample(self._sample(data), current))
        self.samples_since_prediction[key] += 1

        if self.classifier is None:
            ai_result = self._pending("model_not_loaded", len(buffer))
        elif len(buffer) < self.window_size:
            ai_result = self._pending("collecting_window", len(buffer))
        elif key not in self.latest_results or self.samples_since_prediction[key] >= self.step_size:
            window = np.asarray([sample.values for sample in buffer], dtype=np.float32)
            prediction = self.classifier.predict(window)
            accepted, transition_blocked = self._accept_prediction(key, prediction)
            if accepted is None:
                ai_result = self._pending("fallen_requires_evacuating", len(buffer))
                ai_result.update(
                    {
                        "raw_scenario": prediction.scenario,
                        "raw_scenario_ko": LABEL_NAMES_KO.get(
                            prediction.scenario, prediction.scenario
                        ),
                        "raw_confidence": round(prediction.confidence, 4),
                        "transition_blocked": True,
                    }
                )
            else:
                ai_result = {
                    "ready": True,
                    "model": self.classifier.model_name,
                    "scenario": accepted.scenario,
                    "scenario_ko": LABEL_NAMES_KO.get(
                        accepted.scenario, accepted.scenario
                    ),
                    "confidence": round(accepted.confidence, 4),
                    "window_samples": self.window_size,
                    "step_samples": self.step_size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                if transition_blocked:
                    ai_result.update(
                        {
                            "raw_scenario": prediction.scenario,
                            "raw_scenario_ko": LABEL_NAMES_KO.get(
                                prediction.scenario, prediction.scenario
                            ),
                            "raw_confidence": round(prediction.confidence, 4),
                            "transition_blocked": True,
                        }
                    )
            self.latest_results[key] = ai_result
            self.samples_since_prediction[key] = 0
        else:
            ai_result = self.latest_results[key]

        enriched = dict(data)
        enriched["ai"] = dict(ai_result)
        if ai_result.get("ready"):
            enriched["rescue_priority"] = self.priority_engine.evaluate(
                key,
                str(ai_result["scenario"]),
                now=current,
            )
        else:
            enriched["rescue_priority"] = self.priority_engine.pending(
                str(ai_result.get("reason", "ai_pending"))
            )
        return enriched

    def _pending(self, reason: str, count: int) -> dict[str, object]:
        return {
            "ready": False,
            "model": getattr(self.classifier, "model_name", None),
            "scenario": None,
            "confidence": None,
            "reason": reason,
            "samples_collected": count,
            "samples_required": self.window_size,
            "step_samples": self.step_size,
        }
