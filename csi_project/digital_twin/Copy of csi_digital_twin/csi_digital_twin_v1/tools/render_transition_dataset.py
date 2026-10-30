"""Build a recipe-switch dataset with carryover and rejection labels.

This is the next stage after the color-only batch renderer. It renders the
post-switch vials with a deterministic mixture of the previous and current
recipe colors and records the transition risk metadata. Fill-volume defects
are labelled but not rendered because the current USD animation has no
volume parameter.

Example:
    <isaac-sim>/python.sh tools/render_transition_dataset.py \
        --labels-csv data/video5_vial_recipes.csv \
        --out dataset/recipe_transitions --samples-per-vial 4
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

from render_batch1 import (
    ANIMATION_END,
    ANIMATION_START,
    DEFAULT_ISAAC_PYTHON,
    HERE,
    RENDER,
    TANK_COLORS_SRGB,
    positive_int,
)


def read_recipes(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    recipes = []
    for row in rows:
        volumes = {
            tank: float(row[tank]) if row[tank] else 0.0
            for tank in ("t100_ml", "t200_ml", "t400_ml")
        }
        total = float(row["total_ml"])
        if total <= 0:
            raise ValueError(f"vial {row['vial']} must have a positive total_ml")
        color = [
            sum(volumes[tank] * TANK_COLORS_SRGB[tank][channel]
                for tank in volumes) / total
            for channel in range(3)
        ]
        recipes.append({
            "vial": int(row["vial"]),
            "recipe_id": int(row["recipe_id"]),
            "total_ml": total,
            "liquid_color_srgb": color,
        })
    if not recipes:
        raise ValueError("recipe CSV is empty")
    recipes.sort(key=lambda recipe: recipe["vial"])
    return recipes


def color_distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def transition_position(recipes, index):
    recipe_id = recipes[index]["recipe_id"]
    position = 1
    while index - position >= 0 and recipes[index - position]["recipe_id"] == recipe_id:
        position += 1
    return position


def choose_split(transition_id):
    return ("train", "validation", "test")[transition_id % 3]


def build_examples(recipes, samples_per_vial, carryover_tau, volume_tolerance, rng):
    examples = []
    transition_id = -1
    for index, recipe in enumerate(recipes):
        if index == 0 or recipe["recipe_id"] != recipes[index - 1]["recipe_id"]:
            transition_id += 1
        if index == 0:
            continue

        previous = recipes[index - 1]
        position = transition_position(recipes, index)
        carryover = math.exp(-(position - 1) / carryover_tau)
        current_color = recipe["liquid_color_srgb"]
        previous_color = previous["liquid_color_srgb"]
        observed_color = [
            carryover * old + (1.0 - carryover) * new
            for old, new in zip(previous_color, current_color)
        ]
        color_error = color_distance(observed_color, current_color)
        volume_factor = rng.uniform(1.0 - volume_tolerance * 2, 1.0 + volume_tolerance * 2)
        volume_error = abs(volume_factor - 1.0)
        reasons = []
        if color_error > volume_tolerance:
            reasons.append("color_carryover")
        if volume_error > volume_tolerance:
            reasons.append("fill_volume")
        rejected = bool(reasons)
        split = choose_split(transition_id)
        frames = rng.sample(
            range(ANIMATION_START, ANIMATION_END + 1),
            min(samples_per_vial, ANIMATION_END - ANIMATION_START + 1),
        )
        for frame in frames:
            examples.append({
                "vial": recipe["vial"],
                "transition_id": transition_id,
                "recipe_before": previous["recipe_id"],
                "recipe_after": recipe["recipe_id"],
                "vial_index_after_switch": position,
                "frame": frame,
                "expected_color_srgb": current_color,
                "observed_color_srgb": observed_color,
                "rendered_color_srgb": observed_color,
                "expected_volume_ml": recipe["total_ml"],
                "simulated_volume_ml": recipe["total_ml"] * volume_factor,
                "color_error": color_error,
                "volume_error_fraction": volume_error,
                "rejected": rejected,
                "label": "reject" if rejected else "pass",
                "defect": "|".join(reasons),
                "label_quality": "transition_proxy",
                "rendered_recipe_components": ["liquid_color", "color_carryover"],
                "unrendered_recipe_components": ["fill_volume", "dispense_sequence"],
                "split": split,
            })
    rng.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", choices=("FillCam", "CapSealCam"), default="FillCam")
    parser.add_argument("--stage", type=Path, default=HERE / "csi_line.usda")
    parser.add_argument("--samples-per-vial", type=positive_int, default=4)
    parser.add_argument("--carryover-tau", type=float, default=2.0)
    parser.add_argument("--volume-tolerance", type=float, default=0.05)
    parser.add_argument("--subframes", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac-python", type=Path, default=None)
    args = parser.parse_args()
    if args.carryover_tau <= 0 or args.volume_tolerance <= 0:
        parser.error("--carryover-tau and --volume-tolerance must be positive")

    labels_csv = args.labels_csv.expanduser().resolve()
    stage = args.stage.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not labels_csv.is_file():
        parser.error(f"labels CSV does not exist: {labels_csv}")
    if not stage.is_file():
        parser.error(f"stage does not exist: {stage}")
    isaac_python = args.isaac_python or os.environ.get("ISAAC_SIM_PYTHON")
    isaac_python = Path(isaac_python).expanduser().resolve() if isaac_python else DEFAULT_ISAAC_PYTHON
    if not args.dry_run and not isaac_python.exists():
        raise FileNotFoundError(f"Isaac Sim Python not found: {isaac_python}")

    examples = build_examples(
        read_recipes(labels_csv),
        args.samples_per_vial,
        args.carryover_tau,
        args.volume_tolerance,
        random.Random(args.seed),
    )
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, example in enumerate(examples):
            output = image_dir / (
                f"{index:06d}_transition{example['transition_id']:02d}_"
                f"vial{example['vial']:02d}_{example['label']}_frame{example['frame']:04d}.png"
            )
            if not args.dry_run:
                command = [
                    str(isaac_python), str(RENDER), "--camera", args.camera,
                    "--stage", str(stage), "--frame", str(example["frame"]),
                    "--subframes", str(args.subframes), "--out", str(output),
                    "--liquid-color", *(str(channel) for channel in example["rendered_color_srgb"]),
                ]
                subprocess.run(command, check=True, cwd=str(HERE))
            record = {
                **example,
                "image": str(output.relative_to(out)),
                "camera": args.camera,
                "stage": str(stage),
                "seed": args.seed,
            }
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"{'planned' if args.dry_run else 'rendered'} {len(examples)} images")
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
