from __future__ import annotations

from typing import Sequence

import numpy as np


BINARY_CHANNELS = {"status", "motion"}
NONNEGATIVE_CHANNELS = {
    "target_energy",
    "presence_score",
    "distance_mm",
}


def augment_training_windows(
    samples: np.ndarray,
    labels: np.ndarray,
    channels: Sequence[str],
    *,
    copies: int = 0,
    jitter: float = 0.03,
    scale: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if copies < 0:
        raise ValueError("augment_copies는 0 이상이어야 합니다.")
    if jitter < 0:
        raise ValueError("augment_jitter는 0 이상이어야 합니다.")
    if scale < 0:
        raise ValueError("augment_scale은 0 이상이어야 합니다.")
    if samples.ndim != 3:
        raise ValueError("samples는 [윈도우, 시간, 채널] 형태여야 합니다.")
    if len(samples) != len(labels):
        raise ValueError("samples와 labels 개수가 다릅니다.")
    if samples.shape[-1] != len(channels):
        raise ValueError("samples 채널 수와 channels가 다릅니다.")
    if copies == 0:
        return samples, labels

    numeric_indices = [
        index
        for index, channel in enumerate(channels)
        if channel not in BINARY_CHANNELS
    ]
    if not numeric_indices:
        return np.concatenate([samples] * (copies + 1)), np.concatenate(
            [labels] * (copies + 1)
        )

    rng = np.random.default_rng(random_state)
    channel_std = np.std(samples[:, :, numeric_indices], axis=(0, 1))
    channel_scale = np.where(channel_std > 1e-6, channel_std, 1.0)
    augmented_blocks = [samples]

    for _ in range(copies):
        augmented = samples.copy()
        zero_distance_mask = None
        if "distance_mm" in channels:
            distance_index = channels.index("distance_mm")
            zero_distance_mask = samples[:, :, distance_index] == 0
        numeric = augmented[:, :, numeric_indices]
        if scale > 0:
            scale_factors = rng.normal(
                loc=1.0,
                scale=scale,
                size=(len(samples), 1, len(numeric_indices)),
            ).astype(np.float32)
            numeric *= scale_factors
        if jitter > 0:
            noise = rng.normal(
                loc=0.0,
                scale=jitter * channel_scale,
                size=numeric.shape,
            ).astype(np.float32)
            numeric += noise
        augmented[:, :, numeric_indices] = numeric

        for channel_index, channel in enumerate(channels):
            if channel in NONNEGATIVE_CHANNELS:
                augmented[:, :, channel_index] = np.clip(
                    augmented[:, :, channel_index],
                    0.0,
                    None,
                )
        if zero_distance_mask is not None:
            augmented[:, :, distance_index][zero_distance_mask] = 0.0
        augmented_blocks.append(augmented)

    return (
        np.concatenate(augmented_blocks, axis=0),
        np.concatenate([labels] * (copies + 1), axis=0),
    )
