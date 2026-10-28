"""Batch-render synthetic vial-fill training examples from recipe metadata.

The recipe CSV supplies metadata matching the real video dataset. Its source-video
frame columns are deliberately ignored: the USD animation uses time codes 1-686.
Recipe tank volumes are converted to a simple linear color blend (T100 clear,
T200 orange, T400 blue) and applied to the bound Fill_liquid material at render
time. Only color is recipe-conditioned; the USD scene's fill geometry and motion
are generic, so this is not a calibrated optical or volume-accurate simulation.

Run with the Isaac Sim Python executable, for example:

    <isaac-sim>/python.sh tools/render_batch1.py \
        --labels-csv data/video5_vial_recipes.csv \
        --out dataset --samples-per-vial 100 --seed 7
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
RENDER = HERE / "tools" / "render_example.py"
DEFAULT_ISAAC_PYTHON = Path("/isaac-sim/python.sh")
ANIMATION_START = 1
ANIMATION_END = 686
RECIPE_FIELDS = (
    "vial", "recipe_id", "type", "t100_ml", "t200_ml", "t400_ml",
    "total_ml", "dispense_order", "color_purpose",
)
TANK_COLORS_SRGB = {
    "t100_ml": (1.0, 1.0, 1.0),
    "t200_ml": (1.0, 0.28, 0.03),
    "t400_ml": (0.03, 0.20, 1.0),
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def read_vials(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = set(RECIPE_FIELDS)
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    result = []
    for row in rows:
        vial = int(row["vial"])
        recipe_id = int(row["recipe_id"])
        expected_recipe = 5 + (vial - 1) // 3
        if vial < 1 or vial > 30 or recipe_id != expected_recipe:
            raise ValueError(
                f"vial {vial} must map to recipe {expected_recipe}, got {recipe_id}"
            )
        metadata = {field: row[field] for field in RECIPE_FIELDS}
        metadata["vial"] = vial
        metadata["recipe_id"] = recipe_id
        for field in ("t100_ml", "t200_ml", "t400_ml", "total_ml"):
            metadata[field] = float(metadata[field]) if metadata[field] else None
        volume = metadata["total_ml"]
        if volume is None or volume <= 0:
            raise ValueError(f"vial {vial} must have a positive total_ml")
        metadata["liquid_color_srgb"] = [
            sum((metadata[tank] or 0.0) * TANK_COLORS_SRGB[tank][channel]
                for tank in TANK_COLORS_SRGB) / volume
            for channel in range(3)
        ]
        result.append(metadata)

    seen = {row["vial"] for row in result}
    if not seen:
        raise ValueError("recipe CSV is empty")
    if len(seen) != len(result):
        raise ValueError("recipe CSV contains duplicate vial rows")
    if min(seen) < 1 or max(seen) > 30:
        raise ValueError("vial IDs must be within 1..30")
    return result


def choose_frames(vials, samples, rng):
    frames = list(range(ANIMATION_START, ANIMATION_END + 1))
    examples = []
    for vial in vials:
        selected = (
            rng.sample(frames, samples)
            if samples <= len(frames)
            else rng.choices(frames, k=samples)
        )
        vial_position = (vial["vial"] - 1) % 3
        split = ("train", "validation", "test")[vial_position]
        for frame in selected:
            examples.append({
                **vial,
                "frame": frame,
                "label": "pass",
                "defect": "",
                "proxy": False,
                "label_quality": "recipe_color_approximation",
                "color_model": "tank_volume_weighted_srgb_v1",
                "rendered_recipe_components": ["liquid_color"],
                "unrendered_recipe_components": ["fill_volume", "dispense_sequence"],
                "split": split,
            })
    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="video5_vial_recipes.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", choices=("FillCam", "CapSealCam"), default="FillCam")
    parser.add_argument("--stage", type=Path, default=HERE / "csi_line.usda")
    parser.add_argument("--samples-per-vial", type=positive_int, default=100)
    parser.add_argument("--subframes", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=None,
                        help="Path to Isaac Sim python.sh; defaults to /isaac-sim/python.sh")
    args = parser.parse_args()

    isaac_python = args.isaac_python or os.environ.get("ISAAC_SIM_PYTHON")
    if isaac_python is None:
        isaac_python = DEFAULT_ISAAC_PYTHON if DEFAULT_ISAAC_PYTHON.is_file() else Path(sys.executable)
    else:
        isaac_python = Path(isaac_python).expanduser().resolve()
    if not isaac_python.exists():
        raise FileNotFoundError(f"Isaac Sim Python not found: {isaac_python}")

    labels_csv = args.labels_csv.expanduser().resolve()
    stage = args.stage.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not labels_csv.is_file():
        parser.error(f"labels CSV does not exist: {labels_csv}")
    if not stage.is_file():
        parser.error(f"stage does not exist: {stage}")

    examples = choose_frames(
        read_vials(labels_csv), args.samples_per_vial, random.Random(args.seed)
    )
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, example in enumerate(examples):
            name = (
                f"{index:06d}_vial{example['vial']:02d}_"
                f"{example['label']}_frame{example['frame']:04d}.png"
            )
            output = image_dir / name
            command = [
                str(isaac_python), str(RENDER), "--camera", args.camera,
                "--stage", str(stage), "--frame", str(example["frame"]),
                "--subframes", str(args.subframes), "--out", str(output),
                "--liquid-color", *(str(channel) for channel in example["liquid_color_srgb"]),
            ]
            record = {
                **example,
                "image": str(output.relative_to(out)),
                "camera": args.camera,
                "stage": str(stage),
                "seed": args.seed,
            }
            if not args.dry_run:
                print(
                    f"[{index + 1}/{len(examples)}] vial {example['vial']} "
                    f"recipe {example['recipe_id']} frame {example['frame']}",
                    flush=True,
                )
                subprocess.run(command, check=True, cwd=str(HERE))
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
            manifest.flush()

    print(f"{'planned' if args.dry_run else 'rendered'} {len(examples)} images")
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
