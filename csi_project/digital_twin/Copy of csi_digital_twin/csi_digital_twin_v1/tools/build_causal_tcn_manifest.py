"""Build a leakage-safe frame manifest for causal transition models.

The output is JSONL with one record per prediction timestep. Each record
contains a fixed-length window ending at ``frame_index``; padding repeats the
first available frame, so no future frame is exposed to the model. Splits are
assigned by transition/run, never by frame.

Example:
    python tools/build_causal_tcn_manifest.py \
        --dataset-root dataset \
        --output dataset/causal_tcn_manifest.jsonl \
        --window-frames 32 \
        --horizon-frames 10
"""
import argparse
import csv
import json
from pathlib import Path


REQUIRED_LABEL_FIELDS = {
    "filename", "timestamp", "vial", "recipe_id", "pair_id", "scenario",
    "contamination_ml", "recipe_before", "recipe_after", "label",
}


def read_labels(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = REQUIRED_LABEL_FIELDS.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return rows


def as_float(row, field):
    try:
        return float(row[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in label row") from error


def build_records(dataset_root, window_frames, horizon_frames, skip_missing_frames):
    label_files = sorted(dataset_root.glob("**/labels/*.csv"))
    if not label_files:
        raise ValueError(f"no labels/*.csv files found under {dataset_root}")
    loaded = []
    transitions = set()
    for label_path in label_files:
        rows = read_labels(label_path)
        transition_id = f"{rows[0]['recipe_before']}:{rows[0]['recipe_after']}"
        loaded.append((label_path, rows, transition_id))
        transitions.add(transition_id)
    split_names = ("train", "validation", "test")
    transition_splits = {
        transition: split_names[index % len(split_names)]
        for index, transition in enumerate(sorted(transitions))
    }
    records = []
    seen_clip_ids = set()
    for label_path, rows, transition_id in loaded:
        clip_id = f"{label_path.parent.parent.parent.name}/{label_path.stem}"
        if clip_id in seen_clip_ids:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen_clip_ids.add(clip_id)
        split = transition_splits[transition_id]
        frame_dir = label_path.parent.parent / "frames" / label_path.stem
        if not frame_dir.is_dir():
            if skip_missing_frames:
                continue
            raise ValueError(f"missing frame directory: {frame_dir}")
        contamination = as_float(rows[0], "contamination_ml")
        onset = next(
            (index for index, row in enumerate(rows) if row["vial"] == "2"),
            len(rows),
        )
        target_index = max(0, onset + horizon_frames)
        for index, row in enumerate(rows):
            frame_path = frame_dir / row["filename"]
            if not frame_path.is_file():
                raise ValueError(f"missing frame: {frame_path}")
            start = max(0, index - window_frames + 1)
            input_frames = [
                str((frame_dir / rows[past]["filename"]).relative_to(dataset_root))
                for past in range(start, index + 1)
            ]
            input_frames = [input_frames[0]] * (window_frames - len(input_frames)) + input_frames
            records.append({
                "clip_id": clip_id,
                "transition_id": transition_id,
                "split": split,
                "frame_index": index,
                "timestamp_s": as_float(row, "timestamp"),
                "input_frames": input_frames,
                "target_current_reject": int(row["label"] != "pass"),
                "target_future_reject": int(
                    any(rows[past]["label"] != "pass" for past in range(index, min(target_index + 1, len(rows))))
                ),
                "contamination_ml": contamination,
                "contamination_onset_frame": onset,
                "vial": int(row["vial"]),
                "recipe_id": int(row["recipe_id"]),
                "recipe_before": int(row["recipe_before"]),
                "recipe_after": int(row["recipe_after"]),
                "pair_id": row["pair_id"],
                "scenario": row["scenario"],
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--horizon-frames", type=int, default=10)
    parser.add_argument(
        "--skip-missing-frames",
        action="store_true",
        help="skip label files for video-only datasets instead of failing",
    )
    args = parser.parse_args()
    if args.window_frames < 1 or args.horizon_frames < 0:
        parser.error("window-frames must be positive and horizon-frames cannot be negative")
    root = args.dataset_root.expanduser().resolve()
    records = build_records(
        root, args.window_frames, args.horizon_frames, args.skip_missing_frames
    )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().resolve().open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} causal records to {args.output}")


if __name__ == "__main__":
    main()
