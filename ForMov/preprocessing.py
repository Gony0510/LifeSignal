from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    from ForMov.config import CHANNELS, SUPPORTED_LABELS
except ModuleNotFoundError:
    from config import CHANNELS, SUPPORTED_LABELS  # type: ignore


LABEL_ALIASES = {
    "sleeping": "sleeping",
    "sleep": "sleeping",
    "lying": "sleeping",
    "수면": "sleeping",
    "누워있음": "sleeping",
    "누워 있음": "sleeping",
    "evacuating": "evacuating",
    "evacuation": "evacuating",
    "moving": "evacuating",
    "대피": "evacuating",
    "이동": "evacuating",
    "대피중": "evacuating",
    "대피 중": "evacuating",
    "fallen": "fallen",
    "fall": "fallen",
    "collapsed": "fallen",
    "쓰러짐": "fallen",
    "넘어짐": "fallen",
    "no_signal": "no_signal",
    "no-signal": "no_signal",
    "no signal": "no_signal",
    "none": "no_signal",
    "absence": "no_signal",
    "신호없음": "no_signal",
    "신호 없음": "no_signal",
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
            f"지원하지 않는 영상용 라벨입니다: {value!r}. "
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
    if "session_id" not in frame:
        sessions = pd.Series("session_0", index=frame.index, dtype="string")
    else:
        sessions = frame["session_id"].astype("string")
    if "source_session_id" not in frame:
        return sessions.astype(str)
    source = frame["source_session_id"].astype("string").str.strip()
    invalid = source.isna() | source.eq("")
    invalid |= source.str.lower().isin({"nan", "none", "null"})
    return source.mask(invalid, sessions).astype(str)


def to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
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
    return mapped.fillna(pd.to_numeric(series, errors="coerce"))


def prepare_samples(
    frame: pd.DataFrame,
    channels: Sequence[str] = CHANNELS,
) -> pd.DataFrame:
    required = {"session_id", "label", *channels}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))

    prepared = frame.copy()
    prepared["label"] = prepared["label"].map(normalize_label)
    prepared["session_id"] = prepared["session_id"].astype(str)
    prepared["_training_group_id"] = training_group_ids(prepared)
    prepared["_original_order"] = np.arange(len(prepared))

    if "timestamp" in prepared:
        timestamp = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True)
        prepared["_sort_time"] = timestamp.astype("int64", errors="ignore")
        prepared.loc[timestamp.isna(), "_sort_time"] = prepared.loc[
            timestamp.isna(), "_original_order"
        ]
    elif "sample_millis" in prepared:
        prepared["_sort_time"] = pd.to_numeric(
            prepared["sample_millis"], errors="coerce"
        ).fillna(prepared["_original_order"])
    else:
        prepared["_sort_time"] = prepared["_original_order"]

    for channel in channels:
        if channel == "status":
            prepared[channel] = to_binary(prepared[channel])
        else:
            prepared[channel] = pd.to_numeric(prepared[channel], errors="coerce")
    return prepared.sort_values(
        ["session_id", "_sort_time", "_original_order"]
    ).reset_index(drop=True)


def validate_training_data(
    frame: pd.DataFrame,
    channels: Sequence[str] = CHANNELS,
) -> dict[str, object]:
    prepared = prepare_samples(frame, channels)
    labels = set(prepared["label"].unique())
    missing_labels = [label for label in SUPPORTED_LABELS if label not in labels]
    if missing_labels:
        raise ValueError("학습 데이터가 없는 라벨: " + ", ".join(missing_labels))

    by_session = prepared.groupby("session_id")["label"].nunique()
    mixed_sessions = by_session[by_session > 1].index.tolist()
    if mixed_sessions:
        raise ValueError(
            "한 세션에는 하나의 라벨만 있어야 합니다: "
            + ", ".join(map(str, mixed_sessions))
        )

    session_counts = (
        prepared[["_training_group_id", "label"]]
        .drop_duplicates()
        .groupby("label")["_training_group_id"]
        .nunique()
        .to_dict()
    )
    insufficient = [
        label for label in SUPPORTED_LABELS if int(session_counts.get(label, 0)) < 2
    ]
    if insufficient:
        detail = ", ".join(
            f"{label}={int(session_counts.get(label, 0))}세션"
            for label in insufficient
        )
        raise ValueError(
            "세션 누출 없는 학습/검증을 위해 라벨마다 최소 2세션이 필요합니다: "
            + detail
        )
    return {
        "rows": int(len(prepared)),
        "sessions": int(prepared["_training_group_id"].nunique()),
        "session_counts": {
            label: int(session_counts.get(label, 0)) for label in SUPPORTED_LABELS
        },
    }


def build_windows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    channels: Sequence[str] = CHANNELS,
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
        session_labels = session["label"].unique()
        session_groups = session["_training_group_id"].unique()
        if len(session_labels) != 1 or len(session_groups) != 1:
            raise ValueError(f"세션 라벨/원본 세션이 일관되지 않습니다: {session_id}")
        values = session[list(channels)].interpolate(limit_direction="both").fillna(0)
        array = values.to_numpy(dtype=np.float32)
        if len(array) < window_size:
            continue
        for start in range(0, len(array) - window_size + 1, step_size):
            windows.append(array[start : start + window_size])
            labels.append(str(session_labels[0]))
            groups.append(str(session_groups[0]))

    if not windows:
        raise ValueError(
            f"학습 윈도우를 만들 수 없습니다. 세션마다 최소 {window_size}행이 필요합니다."
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
        raise ValueError(f"검증을 위해 최소 {minimum_groups}개 독립 세션이 필요합니다.")
    effective_test_size = max(test_size, len(SUPPORTED_LABELS) / len(unique_groups))
    if effective_test_size >= 1:
        raise ValueError("검증 세트를 분리할 만큼 독립 세션이 충분하지 않습니다.")

    expected = set(SUPPORTED_LABELS)
    for offset in range(300):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=effective_test_size,
            random_state=random_state + offset,
        )
        train_idx, test_idx = next(splitter.split(labels, labels, groups))
        if set(labels[train_idx]) == expected and set(labels[test_idx]) == expected:
            return train_idx, test_idx
    raise ValueError(
        "모든 라벨이 학습/검증 양쪽에 포함되도록 나눌 수 없습니다. "
        "각 라벨의 독립 세션을 늘려주세요."
    )


def _longest_zero_run(values: np.ndarray) -> np.ndarray:
    results = np.zeros(values.shape[0], dtype=np.float32)
    for row_index, row in enumerate(values):
        longest = current = 0
        for value in row:
            if value < 0.5:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        results[row_index] = longest
    return results


def engineer_features(
    samples: np.ndarray,
    channels: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    if samples.ndim != 3:
        raise ValueError("samples는 [윈도우, 시간, 채널] 형태여야 합니다.")
    blocks: list[np.ndarray] = []
    names: list[str] = []
    time_axis = np.arange(samples.shape[1], dtype=np.float32)
    centered_time = time_axis - time_axis.mean()
    denominator = float(np.sum(centered_time**2)) or 1.0
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
            blocks.append(calculator(values)[:, None])
            names.append(f"{channel}_{statistic_name}")
        slopes = np.sum(
            (values - values.mean(axis=1, keepdims=True)) * centered_time,
            axis=1,
        ) / denominator
        blocks.append(slopes[:, None])
        names.append(f"{channel}_slope")

    if "target_energy" in channels:
        energy = samples[:, :, channels.index("target_energy")]
        first_span = energy[:, : min(5, energy.shape[1])].mean(axis=1)
        last_span = energy[:, -min(5, energy.shape[1]) :].mean(axis=1)
        drop = first_span - last_span
        drop_ratio = drop / np.maximum(np.abs(first_span), 1.0)
        extra = (
            ("target_energy_first5_mean", first_span),
            ("target_energy_last5_mean", last_span),
            ("target_energy_drop", drop),
            ("target_energy_drop_ratio", drop_ratio),
        )
        for name, values in extra:
            blocks.append(values[:, None])
            names.append(name)

    if "status" in channels:
        status = samples[:, :, channels.index("status")]
        extras = (
            ("status_active_ratio", np.mean(status >= 0.5, axis=1)),
            ("status_longest_inactive_run", _longest_zero_run(status)),
        )
        for name, values in extras:
            blocks.append(values[:, None])
            names.append(name)
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def fit_channel_scaler(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=(0, 1)).astype(np.float32)
    std = samples.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def scale_sequences(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((samples - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
