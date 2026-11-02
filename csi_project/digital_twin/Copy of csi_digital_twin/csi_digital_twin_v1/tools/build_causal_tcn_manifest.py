"""Build a leakage-safe frame manifest for causal transition models.

The output is JSONL with one record per prediction timestep. Each record
contains a fixed-length window ending at ``frame_index``; padding repeats the
first available frame, so no future frame is exposed to the model. By default,
splits are assigned by render batch, never by frame or clip.

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
    "contamination_ml", "recipe_before", "recipe_after", "label", "render_frame",
}
APPROVED_COLOR_MODEL = "beer_lambert_rgb_v6_video5_r05_r06_opacity_075"
APPROVED_LIQUID_OPACITY = 0.75
APPROVED_RECIPES = {5, 6}


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


def recipe_context(row):
    volumes = tuple(
        float(row[field]) if row[field] else 0.0
        for field in ("t100_ml", "t200_ml", "t400_ml")
    )
    order = tuple(token.strip().upper() for token in row["dispense_order"].split(">") if token.strip())
    if not order:
        raise ValueError("dispense_order cannot be empty")
    colored = [token for token in order if token != "T100"]
    return {
        "recipe_before": int(row["recipe_before"]),
        "recipe_after": int(row["recipe_after"]),
        "tank_volumes_ml": volumes,
        "dispense_order": order,
        "carryover_tank": colored[-1] if colored else None,
    }


def read_color_metadata(dataset_dir):
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        return None, None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    color_model = metadata.get("color_model")
    liquid_opacity = metadata.get("liquid_opacity")
    return color_model, liquid_opacity


def has_approved_color_calibration(rows, color_model, liquid_opacity):
    if color_model != APPROVED_COLOR_MODEL or liquid_opacity != APPROVED_LIQUID_OPACITY:
        return False
    return all(
        int(row["recipe_before"]) in APPROVED_RECIPES
        and int(row["recipe_after"]) in APPROVED_RECIPES
        for row in rows
    )


def assign_group_splits(group_sizes):
    split_names = ("train", "validation", "test")
    target_fractions = {"train": 0.7, "validation": 0.15, "test": 0.15}
    total = sum(group_sizes.values())
    ordered_groups = sorted(group_sizes.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered_groups) < len(split_names):
        raise ValueError("at least three independent groups are required for train/validation/test")
    assigned = {name: 0 for name in split_names}
    group_splits = {}
    for (group, size), split in zip(ordered_groups, split_names):
        group_splits[group] = split
        assigned[split] += size
    for group, size in ordered_groups[len(split_names):]:
        split = max(
            split_names,
            key=lambda name: (
                (target_fractions[name] * total - assigned[name])
                / (target_fractions[name] * total),
                -split_names.index(name),
            )
        )
        group_splits[group] = split
        assigned[split] += size
    return group_splits


def build_records(
    dataset_root,
    window_frames,
    horizon_frames,
    skip_missing_frames,
    split_by="dataset",
    horizon_seconds=None,
    liquid_visible_render_frame=151,
    allow_unverified_color=False,
    prefill_only=False,
):
    label_files = sorted(dataset_root.glob("**/labels/*.csv"))
    if not label_files:
        raise ValueError(f"no labels/*.csv files found under {dataset_root}")
    loaded = []
    group_sizes = {}
    skipped_color_batches = set()
    for label_path in label_files:
        rows = read_labels(label_path)
        dataset_dir = label_path.parents[2]
        color_model, liquid_opacity = read_color_metadata(dataset_dir)
        if not allow_unverified_color and not has_approved_color_calibration(
            rows, color_model, liquid_opacity
        ):
            skipped_color_batches.add(dataset_dir)
            continue
        transition_id = f"{rows[0]['recipe_before']}:{rows[0]['recipe_after']}"
        frame_dir = label_path.parent.parent / "frames" / label_path.stem
        if not frame_dir.is_dir():
            if skip_missing_frames:
                continue
            raise ValueError(f"missing frame directory: {frame_dir}")
        dataset_id = label_path.relative_to(dataset_root).parts[0]
        split_group = transition_id if split_by == "transition" else dataset_id
        loaded.append((label_path, rows, transition_id, split_group, color_model, liquid_opacity))
        group_sizes[split_group] = group_sizes.get(split_group, 0) + len(rows)
    if not loaded:
        raise ValueError(f"no frame-backed labels found under {dataset_root}")
    for dataset_dir in sorted(skipped_color_batches):
        color_model, liquid_opacity = read_color_metadata(dataset_dir)
        print(
            f"skipping unverified color batch {dataset_dir.relative_to(dataset_root)}: "
            f"model={color_model!r}, opacity={liquid_opacity!r}"
        )
    group_splits = assign_group_splits(group_sizes)
    records = []
    seen_clip_ids = set()
    for label_path, rows, transition_id, split_group, color_model, liquid_opacity in loaded:
        clip_id = str(label_path.relative_to(dataset_root).parent.parent / label_path.stem)
        if clip_id in seen_clip_ids:
            raise ValueError(f"duplicate clip id: {clip_id}")
        seen_clip_ids.add(clip_id)
        split = group_splits[split_group]
        frame_dir = label_path.parent.parent / "frames" / label_path.stem
        contamination = as_float(rows[0], "contamination_ml")
        if contamination < 0:
            raise ValueError(f"negative contamination_ml in {label_path}")
        vial2_start = next(
            (index for index, row in enumerate(rows) if row["vial"] == "2"),
            len(rows),
        )
        onset = next(
            (
                index for index, row in enumerate(rows)
                if row["vial"] == "2"
                and as_float(row, "render_frame") >= liquid_visible_render_frame
            ),
            len(rows),
        )
        onset_time = as_float(rows[onset], "timestamp") if onset < len(rows) else None
        context_row = next(
            (candidate for candidate in rows if candidate["vial"] == "2" and candidate["dispense_order"]),
            rows[0],
        )
        context = recipe_context(context_row)
        for index, row in enumerate(rows):
            if prefill_only and index >= vial2_start:
                break
            frame_path = frame_dir / row["filename"]
            if not frame_path.is_file():
                raise ValueError(f"missing frame: {frame_path}")
            start = max(0, index - window_frames + 1)
            input_frames = [
                str((frame_dir / rows[past]["filename"]).relative_to(dataset_root))
                for past in range(start, index + 1)
            ]
            input_frames = [input_frames[0]] * (window_frames - len(input_frames)) + input_frames
            timestamp = as_float(row, "timestamp")
            current_reject = int(contamination > 0 and index >= onset)
            if contamination == 0 or onset >= len(rows):
                future_reject = 0
            elif horizon_seconds is not None:
                future_reject = int(
                    0 < onset_time - timestamp <= horizon_seconds
                )
            else:
                future_reject = int(0 < onset - index <= horizon_frames)
            records.append({
                "clip_id": clip_id,
                "transition_id": transition_id,
                "split": split,
                "frame_index": index,
                "timestamp_s": timestamp,
                "input_frames": input_frames,
                "target_current_reject": current_reject,
                "target_future_reject": future_reject,
                "target_prefill_reject": int(contamination > 0),
                "contamination_ml": contamination,
                "contamination_onset_frame": onset,
                "vial2_start_frame": vial2_start,
                "render_frame": int(as_float(row, "render_frame")),
                "vial": int(row["vial"]),
                "recipe_id": int(row["recipe_id"]),
                "recipe_before": context["recipe_before"],
                "recipe_after": context["recipe_after"],
                "pair_id": row["pair_id"],
                "scenario": row["scenario"],
                "tank_volumes_ml": context["tank_volumes_ml"],
                "dispense_order": context["dispense_order"],
                "carryover_tank": context["carryover_tank"],
                "color_model": color_model,
                "liquid_opacity": liquid_opacity,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument(
        "--horizon-frames",
        type=int,
        default=10,
        help="future prediction horizon in sampled frames (ignored with --horizon-seconds)",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=None,
        help="future prediction horizon in seconds, independent of render FPS",
    )
    parser.add_argument(
        "--split-by",
        choices=("dataset", "transition"),
        default="dataset",
        help="keep entire render batches or recipe directions isolated between splits",
    )
    parser.add_argument(
        "--liquid-visible-render-frame",
        type=int,
        default=16,
        help="animation frame at which the colored vial-2 stream first appears (validated at frame 16)",
    )
    parser.add_argument(
        "--skip-missing-frames",
        action="store_true",
        help="skip label files for video-only datasets instead of failing",
    )
    parser.add_argument(
        "--allow-unverified-color",
        action="store_true",
        help="include batches without the approved recipe 5/6 color calibration",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="emit only records before vial 2 starts filling for pre-fill prediction",
    )
    args = parser.parse_args()
    if (
        args.window_frames < 1
        or args.horizon_frames < 0
        or (args.horizon_seconds is not None and args.horizon_seconds < 0)
        or args.liquid_visible_render_frame < 1
    ):
        parser.error("window-frames must be positive and horizons cannot be negative")
    root = args.dataset_root.expanduser().resolve()
    records = build_records(
        root,
        args.window_frames,
        args.horizon_frames,
        args.skip_missing_frames,
        split_by=args.split_by,
        horizon_seconds=args.horizon_seconds,
        liquid_visible_render_frame=args.liquid_visible_render_frame,
        allow_unverified_color=args.allow_unverified_color,
        prefill_only=args.prefill_only,
    )
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().resolve().open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} causal records to {args.output}")


if __name__ == "__main__":
    main()
