from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

DEFAULT_DATA_PATH = DATA_DIR / "c4001_priority_samples.csv"
DEFAULT_SVM_PATH = ARTIFACT_DIR / "c4001-priority-svm.joblib"
DEFAULT_CNN_PATH = ARTIFACT_DIR / "c4001-priority-cnn.keras"
VPR100_DEFAULT_DATA_PATH = DATA_DIR / "vpr100_priority_samples.csv"
VPR100_DEFAULT_SVM_PATH = ARTIFACT_DIR / "vpr100-priority-svm.joblib"
VPR100_DEFAULT_CNN_PATH = ARTIFACT_DIR / "vpr100-priority-cnn.keras"
DEFAULT_HTML_PATH = BASE_DIR / "LifeSignal_ForMov.html"

SUPPORTED_LABELS = ("sleeping", "evacuating", "fallen", "no_signal")
LABEL_NAMES_KO = {
    "sleeping": "누워 있음",
    "evacuating": "대피 중",
    "fallen": "쓰러짐",
    "no_signal": "신호 없음",
}

CHANNELS = ("target_energy", "status")
VPR100_CHANNELS = ("presence_score", "distance_mm", "status")
SAMPLE_INTERVAL_SEC = 0.2
WINDOW_SIZE = 25
STEP_SIZE = 5
SLEEPING_DANGER_AFTER_SEC = 5.0
DANGER_HOLD_SEC = 4.0

CSV_COLUMNS = (
    "session_id",
    "label",
    "timestamp",
    "sample_millis",
    "sensor_id",
    "room",
    "location",
    "status",
    "target_energy",
    "source_session_id",
    "is_augmented",
    "augmentation_id",
)

VPR100_CSV_COLUMNS = (
    "session_id",
    "label",
    "timestamp",
    "sensor",
    "room",
    "location",
    "status",
    "presence_score",
    "distance_mm",
    "source_session_id",
    "is_augmented",
    "augmentation_id",
)
