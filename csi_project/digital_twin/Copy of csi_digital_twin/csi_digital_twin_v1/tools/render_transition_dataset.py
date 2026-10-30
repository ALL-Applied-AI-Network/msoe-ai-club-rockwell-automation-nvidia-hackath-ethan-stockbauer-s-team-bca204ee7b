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

NEXT_FILL_START = 21
EXPECTED_RECIPES = {
    5: (10, 10, 10, 30, ("t200", "t400", "t100")),
    6: (30, 10, 0, 40, ("t200", "t100")),
    7: (20, 0, 50, 70, ("t400", "t100")),
    8: (20, 20, 20, 60, ("t200", "t400", "t100")),
    9: (10, 35, 35, 80, ("t200", "t400", "t100")),
    10: (45, 15, 15, 75, ("t200", "t400", "t100")),
    11: (10, 10, 50, 70, ("t200", "t400", "t100")),
    12: (10, 50, 10, 70, ("t200", "t400", "t100")),
    13: (0, 50, 20, 70, ("t200", "t400")),
    14: (60, 0, 10, 70, ("t400", "t100")),
}


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
        dispense_order = tuple(
            tank.strip().lower()
            for tank in row.get("dispense_order", "").split(">")
            if tank.strip()
        )
        expected = EXPECTED_RECIPES.get(int(row["recipe_id"]))
        actual = (
            volumes["t100_ml"],
            volumes["t200_ml"],
            volumes["t400_ml"],
            total,
            dispense_order,
        )
        if expected is None or actual != expected:
            raise ValueError(
                f"recipe {row['recipe_id']} does not match the supported tank recipe: "
                f"expected {expected}, got {actual}"
            )
        recipes.append({
            "vial": int(row["vial"]),
            "recipe_id": int(row["recipe_id"]),
            "total_ml": total,
            "liquid_color_srgb": color,
            "dispense_order": dispense_order,
            "tank_volumes_ml": volumes,
        })
    if not recipes:
        raise ValueError("recipe CSV is empty")
    recipes.sort(key=lambda recipe: recipe["vial"])
    return recipes


def dispensing_tank_color(recipe, frame):
    """Return the tank color for the volume-proportional dispense segment."""
    order = recipe["dispense_order"]
    if not order:
        return recipe["liquid_color_srgb"]
    tank_volumes = recipe["tank_volumes_ml"]
    elapsed = min(
        ANIMATION_END - ANIMATION_START,
        max(0, frame - ANIMATION_START),
    )
    progress_ml = (elapsed / (ANIMATION_END - ANIMATION_START)) * recipe["total_ml"]
    accumulated_ml = 0.0
    for tank in order:
        tank_key = f"{tank}_ml"
        try:
            tank_volume = tank_volumes[tank_key]
            tank_color = TANK_COLORS_SRGB[tank_key]
        except KeyError as error:
            raise ValueError(
                f"unknown dispense tank {tank_key!r} in recipe {recipe['recipe_id']}"
            ) from error
        accumulated_ml += tank_volume
        if progress_ml < accumulated_ml:
            return list(tank_color)
    return list(TANK_COLORS_SRGB[f"{order[-1]}_ml"])


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


def build_examples(
    recipes,
    transition_frames,
    nozzle_residual_frames,
    carryover_tau,
    volume_tolerance,
    scenario,
    rng,
):
    examples = []
    transition_id = -1
    for index, recipe in enumerate(recipes):
        if index == 0 or recipe["recipe_id"] != recipes[index - 1]["recipe_id"]:
            transition_id += 1
        if index == 0 or recipe["recipe_id"] == recipes[index - 1]["recipe_id"]:
            continue

        previous = recipes[index - 1]
        current_color = recipe["liquid_color_srgb"]
        previous_color = previous["liquid_color_srgb"]
        position = transition_position(recipes, index)
        volume_factor = rng.uniform(1.0 - volume_tolerance * 2, 1.0 + volume_tolerance * 2)
        variants = ("clean", "residual") if scenario == "both" else (scenario,)
        for variant_name in variants:
            for sequence_index in range(transition_frames * 2):
                is_next = sequence_index >= transition_frames
                frame = (
                    ANIMATION_END - transition_frames + sequence_index + 1
                    if not is_next
                    else sequence_index - transition_frames + NEXT_FILL_START
                )
                next_index = sequence_index - transition_frames
                residual_active = (
                    variant_name == "residual"
                    and is_next
                    and next_index < nozzle_residual_frames
                )
                carryover = (
                    math.exp(-next_index / carryover_tau)
                    if variant_name == "residual" and is_next
                    else 0.0
                )
                observed_color = (
                    [
                        carryover * old + (1.0 - carryover) * new
                        for old, new in zip(previous_color, current_color)
                    ]
                    if is_next
                    else previous_color
                )
                color_error = color_distance(observed_color, current_color)
                volume_error = abs(volume_factor - 1.0) if is_next else 0.0
                reasons = []
                if color_error > volume_tolerance:
                    reasons.append("color_carryover")
                if volume_error > volume_tolerance:
                    reasons.append("fill_volume")
                stream_recipe = previous if residual_active or not is_next else recipe
                examples.append({
                    "scenario": variant_name,
                    "sequence_index": sequence_index,
                    "phase": "previous_fill" if not is_next else "next_fill",
                    "vial": recipe["vial"],
                    "transition_id": transition_id,
                    "recipe_before": previous["recipe_id"],
                    "recipe_after": recipe["recipe_id"],
                    "vial_index_after_switch": position,
                    "frame": frame,
                    "expected_color_srgb": current_color,
                    "observed_color_srgb": observed_color,
                    "rendered_color_srgb": observed_color,
                    "stream_color_srgb": dispensing_tank_color(stream_recipe, frame),
                    "expected_volume_ml": recipe["total_ml"],
                    "simulated_volume_ml": recipe["total_ml"] * volume_factor,
                    "color_error": color_error,
                    "volume_error_fraction": volume_error,
                    "rejected": bool(reasons),
                    "label": "reject" if reasons else "pass",
                    "defect": "|".join(reasons),
                    "label_quality": "transition_proxy",
                    "rendered_recipe_components": ["liquid_color", "color_carryover"],
                    "unrendered_recipe_components": ["fill_volume"],
                    "split": choose_split(transition_id),
                    "nozzle_residual": residual_active,
                })
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", choices=("FillCam", "CapSealCam"), default="FillCam")
    parser.add_argument("--stage", type=Path, default=HERE / "csi_line.usda")
    parser.add_argument("--transition-frames", type=positive_int, default=32,
                        help="contiguous frames before and after each recipe changeover")
    parser.add_argument("--nozzle-residual-frames", type=positive_int, default=4,
                        help="next-recipe frames that retain the previous tank in residual scenarios")
    parser.add_argument("--scenario", choices=("clean", "residual", "both"), default="both")
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
        args.transition_frames,
        args.nozzle_residual_frames,
        args.carryover_tau,
        args.volume_tolerance,
        args.scenario,
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
                    "--stream-color", *(str(channel) for channel in example["stream_color_srgb"]),
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
