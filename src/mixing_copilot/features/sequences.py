from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalSequences:
    values: np.ndarray
    valid_mask: np.ndarray
    vial_ids: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]


def build_causal_sequences(
    frame_data: pd.DataFrame,
    feature_names: list[str],
    history: int = 8,
) -> CausalSequences:
    """Create left-padded histories ending at each row; no future row is included."""
    if history < 1:
        raise ValueError("history must be positive")
    required = {"vial_id", "corrected_media_seconds", *feature_names}
    missing = required.difference(frame_data.columns)
    if missing:
        raise ValueError(f"missing sequence columns: {', '.join(sorted(missing))}")
    ordered = frame_data.sort_values(["vial_id", "corrected_media_seconds"]).reset_index(drop=True)
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    vial_ids: list[int] = []
    timestamps: list[float] = []
    for vial_id, group in ordered.groupby("vial_id", sort=False):
        values = group[feature_names].astype(float).fillna(0).to_numpy()
        for index in range(len(values)):
            start = max(0, index - history + 1)
            prefix = values[start : index + 1]
            pad = history - len(prefix)
            sequences.append(np.pad(prefix, ((pad, 0), (0, 0))))
            masks.append(np.asarray([False] * pad + [True] * len(prefix)))
            vial_ids.append(int(vial_id))
            timestamps.append(float(group.iloc[index]["corrected_media_seconds"]))
    return CausalSequences(
        values=np.asarray(sequences, dtype=np.float32),
        valid_mask=np.asarray(masks, dtype=bool),
        vial_ids=np.asarray(vial_ids),
        timestamps=np.asarray(timestamps),
        feature_names=feature_names,
    )
