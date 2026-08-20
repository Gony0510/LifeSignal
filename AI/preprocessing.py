from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_CHANNELS = (
    "presence_score",
    "distance_mm",
    "motion",
    "status",
)
SUPPORTED_LABELS = ("human", "empty", "dog")
LABEL_ALIASES = {
    "human": "human",
    "person": "human",
    "사람": "human",
    "empty": "empty",
    "none": "empty",
    "no_target": "empty",
    "no target": "empty",
    "absence": "empty",
    "빈 공간": "empty",
    "빈공간": "empty",
    "아무것도없는상태": "empty",
    "아무것도 없는 상태": "empty",
    "빈 상태": "empty",
    "없음": "empty",
    "dog": "dog",
    "개": "dog",
    # 기존 2개 라벨로 수집한 파일을 계속 사용할 수 있게 합니다.
    "pet": "dog",
    "animal": "dog",
    "반려동물": "dog",
}


@dataclass(frozen=True)
class WindowDataset:
    samples: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    channels: tuple[str, ...]
    window_size: int


def normalize_label(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in LABEL_ALIASES:
        raise ValueError(
            f"지원하지 않는 라벨입니다: {value!r}. "
            f"사용 가능 라벨: {', '.join(SUPPORTED_LABELS)}"
        )
    return LABEL_ALIASES[normalized]


def load_sensor_csv(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"센서 데이터 파일을 찾을 수 없습니다: {path}")
        frame = pd.read_csv(path)
        frame["_source_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("학습할 CSV 파일을 한 개 이상 지정해야 합니다.")
    return pd.concat(frames, ignore_index=True)


def training_group_ids(frame: pd.DataFrame) -> pd.Series:
    """학습/검증 분리에 사용할 실제 수집 세션 ID를 반환합니다.

    CSV에 저장한 증강 세션은 별도의 ``session_id``를 사용하지만,
    ``source_session_id``가 같은 데이터는 같은 실제 수집 세션에서 파생된
    것이므로 반드시 같은 학습/검증 그룹에 남겨야 합니다.
    """

    if "session_id" not in frame:
        session_ids = pd.Series("session_0", index=frame.index, dtype="string")
    else:
        session_ids = frame["session_id"].astype("string")

    if "source_session_id" not in frame:
        return session_ids.astype(str)

    source_ids = frame["source_session_id"].astype("string").str.strip()
    invalid_source = source_ids.isna() | source_ids.eq("")
    invalid_source |= source_ids.str.lower().isin({"nan", "none", "null"})
    return source_ids.mask(invalid_source, session_ids).astype(str)


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1.0,
            "1": 1.0,
            "yes": 1.0,
            "false": 0.0,
            "0": 0.0,
            "no": 0.0,
            "none": 0.0,
            "nan": np.nan,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def prepare_samples(
    frame: pd.DataFrame,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> pd.DataFrame:
    required = {"label", *channels}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {', '.join(missing)}")

    prepared = frame.copy()
    prepared["label"] = prepared["label"].map(normalize_label)
    if "session_id" not in prepared:
        prepared["session_id"] = "session_0"
    prepared["session_id"] = prepared["session_id"].astype(str)
    prepared["_training_group_id"] = training_group_ids(prepared)

    if "timestamp" in prepared:
        prepared["_sort_time"] = pd.to_datetime(
            prepared["timestamp"],
            errors="coerce",
            utc=True,
        )
    else:
        prepared["_sort_time"] = np.arange(len(prepared))

    for channel in channels:
        if channel in {"motion", "status"}:
            prepared[channel] = _to_binary(prepared[channel])
        else:
            prepared[channel] = pd.to_numeric(prepared[channel], errors="coerce")

    return prepared.sort_values(["session_id", "_sort_time"]).reset_index(drop=True)


def build_windows(
    frame: pd.DataFrame,
    *,
    window_size: int = 40,
    step_size: int = 10,
    channels: Sequence[str] = DEFAULT_CHANNELS,
) -> WindowDataset:
    if window_size < 2:
        raise ValueError("window_size는 2 이상이어야 합니다.")
    if step_size < 1:
        raise ValueError("step_size는 1 이상이어야 합니다.")

    prepared = prepare_samples(frame, channels)
    windows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []

    for session_id, session in prepared.groupby("session_id", sort=False):
        unique_labels = session["label"].unique()
        if len(unique_labels) != 1:
            raise ValueError(
                f"한 세션에는 하나의 라벨만 있어야 합니다: {session_id} "
                f"({', '.join(unique_labels)})"
            )
        unique_groups = session["_training_group_id"].unique()
        if len(unique_groups) != 1:
            raise ValueError(
                "한 세션에는 하나의 source_session_id만 있어야 합니다: "
                f"{session_id} ({', '.join(unique_groups)})"
            )

        values = session[list(channels)].copy()
        values = values.interpolate(limit_direction="both").fillna(0.0)
        array = values.to_numpy(dtype=np.float32)
        if len(array) < window_size:
            continue

        for start in range(0, len(array) - window_size + 1, step_size):
            windows.append(array[start : start + window_size])
            labels.append(unique_labels[0])
            groups.append(str(unique_groups[0]))

    if not windows:
        raise ValueError(
            "학습 윈도우를 만들 수 없습니다. 각 세션에 최소 "
            f"{window_size}개의 측정값이 필요합니다."
        )

    return WindowDataset(
        samples=np.asarray(windows, dtype=np.float32),
        labels=np.asarray(labels),
        groups=np.asarray(groups),
        channels=tuple(channels),
        window_size=window_size,
    )


def grouped_train_test_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    minimum_groups = len(SUPPORTED_LABELS) * 2
    if len(unique_groups) < minimum_groups:
        raise ValueError(
            "세션 누출 없는 검증을 위해 최소 "
            f"{minimum_groups}개 세션이 필요합니다. "
            "사람, 빈 공간, 개를 각각 2세션 이상 수집하세요."
        )

    for offset in range(100):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + offset,
        )
        train_idx, test_idx = next(splitter.split(labels, labels, groups))
        if (
            len(np.unique(labels[train_idx])) == len(SUPPORTED_LABELS)
            and len(np.unique(labels[test_idx])) == len(SUPPORTED_LABELS)
        ):
            return train_idx, test_idx

    raise ValueError(
        "사람, 빈 공간, 개 세션이 학습/검증 양쪽에 들어가도록 "
        "나눌 수 없습니다. "
        "각 라벨의 독립 세션 수를 늘려주세요."
    )


def engineer_features(
    samples: np.ndarray,
    channels: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    if samples.ndim != 3:
        raise ValueError("samples는 [윈도우, 시간, 채널] 형태여야 합니다.")

    feature_blocks: list[np.ndarray] = []
    names: list[str] = []
    time_axis = np.arange(samples.shape[1], dtype=np.float32)
    centered_time = time_axis - time_axis.mean()
    time_denominator = float(np.sum(centered_time**2)) or 1.0

    statistics = (
        ("mean", lambda x: np.mean(x, axis=1)),
        ("std", lambda x: np.std(x, axis=1)),
        ("min", lambda x: np.min(x, axis=1)),
        ("max", lambda x: np.max(x, axis=1)),
        ("median", lambda x: np.median(x, axis=1)),
        ("q25", lambda x: np.quantile(x, 0.25, axis=1)),
        ("q75", lambda x: np.quantile(x, 0.75, axis=1)),
        ("range", lambda x: np.ptp(x, axis=1)),
        ("first", lambda x: x[:, 0]),
        ("last", lambda x: x[:, -1]),
        ("mean_abs_diff", lambda x: np.mean(np.abs(np.diff(x, axis=1)), axis=1)),
        ("std_diff", lambda x: np.std(np.diff(x, axis=1), axis=1)),
    )

    for channel_index, channel in enumerate(channels):
        values = samples[:, :, channel_index]
        for statistic_name, calculator in statistics:
            feature_blocks.append(calculator(values)[:, None])
            names.append(f"{channel}_{statistic_name}")

        slopes = np.sum(
            (values - values.mean(axis=1, keepdims=True)) * centered_time,
            axis=1,
        ) / time_denominator
        feature_blocks.append(slopes[:, None])
        names.append(f"{channel}_slope")

    if "presence_score" in channels and "distance_mm" in channels:
        score = samples[:, :, channels.index("presence_score")]
        distance = samples[:, :, channels.index("distance_mm")]
        correlations = np.zeros(len(samples), dtype=np.float32)
        for index in range(len(samples)):
            if np.std(score[index]) > 0 and np.std(distance[index]) > 0:
                correlations[index] = np.corrcoef(score[index], distance[index])[0, 1]
        feature_blocks.append(np.nan_to_num(correlations)[:, None])
        names.append("presence_distance_correlation")

    return np.concatenate(feature_blocks, axis=1).astype(np.float32), names


def fit_channel_scaler(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=(0, 1)).astype(np.float32)
    std = samples.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def scale_sequences(
    samples: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((samples - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )
