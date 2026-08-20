from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from AI.training_presets import (
        C4001_PRESET,
        VPR100_PRESET,
        SensorTrainingPreset,
    )
except ModuleNotFoundError:
    from training_presets import (  # type: ignore
        C4001_PRESET,
        VPR100_PRESET,
        SensorTrainingPreset,
    )


PROVENANCE_COLUMNS = (
    "source_session_id",
    "is_augmented",
    "augmentation_id",
)
SENSOR_PRESETS = {
    "vpr100": VPR100_PRESET,
    "c4001": C4001_PRESET,
}


@dataclass(frozen=True)
class CsvAugmentationResult:
    data_path: Path
    sensor_type: str
    source_sessions: tuple[str, ...]
    source_rows: int
    generated_rows: int
    total_rows: int
    applied: bool
    backup_path: Path | None
    generated_session_ids: tuple[str, ...]
    generated_rows_by_source: tuple[tuple[str, int], ...]


def _augmentation_flags(frame: pd.DataFrame) -> pd.Series:
    if "is_augmented" not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame["is_augmented"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _with_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    session_ids = prepared["session_id"].astype(str)

    if "source_session_id" not in prepared:
        prepared["source_session_id"] = session_ids
    else:
        source_ids = prepared["source_session_id"].astype("string").str.strip()
        invalid_source = source_ids.isna() | source_ids.eq("")
        invalid_source |= source_ids.str.lower().isin({"nan", "none", "null"})
        prepared["source_session_id"] = source_ids.mask(
            invalid_source,
            session_ids,
        )

    prepared["is_augmented"] = _augmentation_flags(prepared)
    if "augmentation_id" not in prepared:
        prepared["augmentation_id"] = ""
    else:
        prepared["augmentation_id"] = (
            prepared["augmentation_id"].astype("string").fillna("")
        )
    return prepared


def _validate_options(count: int, jitter: float, scale: float) -> None:
    if count < 1:
        raise ValueError("count는 1 이상이어야 합니다.")
    if jitter < 0:
        raise ValueError("jitter는 0 이상이어야 합니다.")
    if scale < 0:
        raise ValueError("scale은 0 이상이어야 합니다.")


def _allocate_row_counts(
    session_sizes: list[int],
    total_count: int,
) -> list[int]:
    sizes = np.asarray(session_sizes, dtype=np.float64)
    if len(sizes) == 0 or np.any(sizes <= 0):
        raise ValueError("증강할 원본 세션에 데이터가 없습니다.")

    exact = sizes / sizes.sum() * total_count
    allocated = np.floor(exact).astype(np.int64)
    remaining = int(total_count - allocated.sum())
    if remaining:
        fractional = exact - allocated
        order = np.argsort(-fractional, kind="stable")
        allocated[order[:remaining]] += 1
    return allocated.astype(int).tolist()


def _positive_median_step(values: np.ndarray, fallback: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return fallback
    positive = np.diff(finite)
    positive = positive[positive > 0]
    if len(positive) == 0:
        return fallback
    return float(np.median(positive))


def _rebuild_sequence_columns(
    augmented: pd.DataFrame,
    source: pd.DataFrame,
    *,
    start_index: int,
    existing_augmented: pd.DataFrame,
) -> None:
    row_offsets = np.arange(len(augmented), dtype=np.float64)

    if "sample_millis" in source:
        millis = pd.to_numeric(source["sample_millis"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        millis_step = _positive_median_step(millis, 200.0)
        existing_millis = pd.to_numeric(
            existing_augmented.get("sample_millis", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        if not existing_millis.empty:
            millis_start = float(existing_millis.max()) + millis_step
        else:
            millis_start = (
                millis[start_index] if np.isfinite(millis[start_index]) else 0.0
            )
        augmented["sample_millis"] = np.rint(
            millis_start + row_offsets * millis_step
        ).astype(np.int64)

    if "timestamp" not in source:
        return

    numeric_timestamps = pd.to_numeric(source["timestamp"], errors="coerce")
    if numeric_timestamps.notna().all():
        numeric = numeric_timestamps.to_numpy(dtype=np.float64)
        timestamp_step = _positive_median_step(numeric, 1.0)
        existing_numeric = pd.to_numeric(
            existing_augmented.get("timestamp", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        if not existing_numeric.empty:
            timestamp_start = float(existing_numeric.max()) + timestamp_step
        else:
            timestamp_start = numeric[start_index]
        rebuilt = timestamp_start + row_offsets * timestamp_step
        if np.allclose(rebuilt, np.rint(rebuilt)):
            rebuilt = np.rint(rebuilt).astype(np.int64)
        augmented["timestamp"] = rebuilt
        return

    parsed = pd.to_datetime(source["timestamp"], errors="coerce", utc=True)
    valid = parsed.dropna()
    if valid.empty:
        augmented["timestamp"] = row_offsets
        return

    valid_nanoseconds = valid.astype("int64").to_numpy(dtype=np.float64)
    interval_nanoseconds = _positive_median_step(
        valid_nanoseconds,
        200_000_000.0,
    )
    start_timestamp = parsed.iloc[start_index]
    existing_timestamps = pd.to_datetime(
        existing_augmented.get("timestamp", pd.Series(dtype="string")),
        errors="coerce",
        utc=True,
    ).dropna()
    if not existing_timestamps.empty:
        start_timestamp = existing_timestamps.max() + pd.to_timedelta(
            interval_nanoseconds,
            unit="ns",
        )
    elif pd.isna(start_timestamp):
        start_timestamp = valid.iloc[0]
    augmented["timestamp"] = [
        (
            start_timestamp
            + pd.to_timedelta(offset * interval_nanoseconds, unit="ns")
        ).isoformat()
        for offset in row_offsets
    ]


def build_augmented_dataset(
    frame: pd.DataFrame,
    preset: SensorTrainingPreset,
    *,
    sessions: list[str],
    count: int,
    jitter: float = 0.03,
    scale: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, tuple[str, ...], int]:
    _validate_options(count, jitter, scale)
    if not sessions:
        raise ValueError("증강할 원본 세션을 한 개 이상 지정해야 합니다.")
    if len(set(sessions)) != len(sessions):
        raise ValueError("sessions에 같은 세션이 중복되어 있습니다.")

    required = {"session_id", "label", *preset.channels}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))

    prepared = _with_provenance(frame)
    original_rows = prepared.loc[~prepared["is_augmented"]].copy()
    available_sessions = set(original_rows["session_id"].astype(str))
    missing_sessions = [session for session in sessions if session not in available_sessions]
    if missing_sessions:
        raise ValueError(
            "원본 데이터에서 세션을 찾을 수 없습니다: "
            + ", ".join(missing_sessions)
            + ". 이미 증강된 세션은 다시 증강할 수 없습니다."
        )

    rng = np.random.default_rng(random_state)
    augmented_frames: list[pd.DataFrame] = []
    generated_session_ids: list[str] = []
    sources: list[pd.DataFrame] = []

    for source_session_id in sessions:
        source = original_rows.loc[
            original_rows["session_id"].astype(str).eq(source_session_id)
        ].copy()
        labels = source["label"].astype(str).str.strip().unique()
        if len(labels) != 1:
            raise ValueError(
                "한 원본 세션에는 하나의 라벨만 있어야 합니다: "
                f"{source_session_id}"
            )

        sources.append(source)

    allocations = _allocate_row_counts(
        [len(source) for source in sources],
        count,
    )

    for source_session_id, source, allocated_count in zip(
        sessions,
        sources,
        allocations,
        strict=True,
    ):
        if allocated_count == 0:
            continue

        source_group_ids = source["source_session_id"].astype(str).unique()
        if len(source_group_ids) != 1:
            raise ValueError(
                "한 원본 세션의 촬영 그룹이 일관되지 않습니다: "
                f"{source_session_id}"
            )
        source_group_id = str(source_group_ids[0])

        start_index = int(rng.integers(0, len(source)))
        source_indices = (start_index + np.arange(allocated_count)) % len(source)
        augmented = source.iloc[source_indices].copy().reset_index(drop=True)

        numeric_channels = [
            channel
            for channel in preset.channels
            if channel not in {"status", "motion"}
        ]
        for channel in numeric_channels:
            values = pd.to_numeric(source[channel], errors="coerce")
            invalid_count = int(values.isna().sum())
            if invalid_count:
                raise ValueError(
                    f"{source_session_id}의 {channel}에 숫자가 아닌 값이 "
                    f"{invalid_count}개 있습니다."
                )

            numeric = values.to_numpy(dtype=np.float64)
            signal_std = float(np.std(numeric))
            noise_base = signal_std if signal_std > 1e-6 else 1.0
            selected_numeric = numeric[source_indices]
            zero_mask = selected_numeric == 0

            scale_factor = (
                float(rng.normal(1.0, scale)) if scale > 0 else 1.0
            )
            transformed = selected_numeric * scale_factor
            if jitter > 0:
                transformed += rng.normal(
                    0.0,
                    jitter * noise_base,
                    size=allocated_count,
                )
            transformed = np.rint(
                np.clip(transformed, 0.0, None)
            ).astype(np.int64)
            transformed[zero_mask] = 0
            augmented[channel] = transformed

        generated_session_id = f"{source_session_id}__aug"
        existing_augmented = prepared.loc[
            prepared["is_augmented"]
            & prepared["session_id"].astype(str).eq(generated_session_id)
        ]

        _rebuild_sequence_columns(
            augmented,
            source,
            start_index=start_index,
            existing_augmented=existing_augmented,
        )
        augmented["session_id"] = generated_session_id
        augmented["source_session_id"] = source_group_id
        augmented["is_augmented"] = True
        augmented["augmentation_id"] = f"aug:{source_session_id}"
        augmented_frames.append(augmented)
        generated_session_ids.append(generated_session_id)

    augmented_rows = pd.concat(augmented_frames, ignore_index=True)
    combined = pd.concat([prepared, augmented_rows], ignore_index=True)
    return combined, tuple(generated_session_ids), int(sum(map(len, sources)))


def augment_csv_file(
    data_path: str | Path,
    preset: SensorTrainingPreset,
    *,
    sessions: list[str],
    count: int,
    jitter: float = 0.03,
    scale: float = 0.05,
    random_state: int = 42,
    apply: bool = False,
) -> CsvAugmentationResult:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"센서 CSV 파일을 찾을 수 없습니다: {path}")

    frame = pd.read_csv(path)
    combined, generated_session_ids, source_rows = build_augmented_dataset(
        frame,
        preset,
        sessions=sessions,
        count=count,
        jitter=jitter,
        scale=scale,
        random_state=random_state,
    )
    generated_rows = len(combined) - len(frame)
    backup_path: Path | None = None

    if apply:
        backup_dir = path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_pattern = re.compile(
            rf"^{re.escape(path.stem)}_backup_(\d+){re.escape(path.suffix)}$"
        )
        used_backup_numbers = []
        for candidate in backup_dir.iterdir():
            match = backup_pattern.match(candidate.name)
            if match:
                used_backup_numbers.append(int(match.group(1)))
        backup_number = max(used_backup_numbers, default=0) + 1
        backup_path = backup_dir / (
            f"{path.stem}_backup_{backup_number:03d}{path.suffix}"
        )
        shutil.copy2(path, backup_path)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            combined.to_csv(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return CsvAugmentationResult(
        data_path=path,
        sensor_type=preset.sensor_type,
        source_sessions=tuple(sessions),
        source_rows=source_rows,
        generated_rows=generated_rows,
        total_rows=len(combined),
        applied=apply,
        backup_path=backup_path,
        generated_session_ids=generated_session_ids,
        generated_rows_by_source=tuple(
            zip(
                sessions,
                _allocate_row_counts(
                    [
                        int((frame["session_id"].astype(str) == session).sum())
                        for session in sessions
                    ],
                    count,
                ),
                strict=True,
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V-PR100 또는 C4001 CSV에 원본 세션 기반 증강 데이터를 추가합니다. "
            "--apply가 없으면 파일을 변경하지 않고 예상 결과만 표시합니다."
        )
    )
    parser.add_argument("--sensor", choices=sorted(SENSOR_PRESETS), required=True)
    parser.add_argument("--data", help="대상 CSV 경로(생략 시 센서 기본 파일)")
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="증강할 원본 세션 ID 목록",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="CSV에 추가할 증강 데이터의 정확한 총 행 수",
    )
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="백업을 만든 뒤 증강 데이터를 실제 CSV에 추가",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = SENSOR_PRESETS[args.sensor]
    data_path = args.data or preset.default_data
    try:
        result = augment_csv_file(
            data_path,
            preset,
            sessions=args.sessions,
            count=args.count,
            jitter=args.jitter,
            scale=args.scale,
            random_state=args.random_state,
            apply=args.apply,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"어그멘테이션 실패: {exc}") from exc

    mode = "적용 완료" if result.applied else "미리보기"
    print(f"\n[{mode}] {preset.display_name} CSV 어그멘테이션")
    print(f"대상 파일: {result.data_path}")
    print(f"원본 세션: {', '.join(result.source_sessions)}")
    print(f"참조한 원본 행: {result.source_rows:,}개")
    print(
        "세션별 생성 행: "
        + ", ".join(
            f"{session}={row_count:,}개"
            for session, row_count in result.generated_rows_by_source
        )
    )
    print(f"추가될 행: {result.generated_rows:,}개")
    print(f"완료 후 전체 행: {result.total_rows:,}개")
    if result.backup_path is not None:
        print(f"원본 백업: {result.backup_path}")
    if not result.applied:
        print("파일은 변경하지 않았습니다. 실제 적용하려면 --apply를 추가하세요.")


if __name__ == "__main__":
    main()
