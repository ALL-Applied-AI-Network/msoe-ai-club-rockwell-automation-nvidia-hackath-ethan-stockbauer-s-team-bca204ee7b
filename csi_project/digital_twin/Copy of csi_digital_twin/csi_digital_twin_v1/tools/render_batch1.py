"""Batch-render synthetic vial-fill training examples from recipe metadata.

The recipe CSV supplies metadata matching the real video dataset. Its source-video
frame columns are deliberately ignored: the USD animation uses time codes 1-686.
Recipe tank volumes are converted to an approximate transparent-dye color blend
and applied to the bound Fill_liquid material at render time. Clear T100 does
not alter the visible color of the dyed liquid; colored liquids combine by
channel-wise optical density. Only color is recipe-conditioned; the USD scene's
fill geometry and motion are generic, so this is not a calibrated optical or
volume-accurate simulation.

Run with the Isaac Sim Python executable, for example:

    <isaac-sim>/python.sh tools/render_batch1.py \
        --labels-csv data/video5_vial_recipes.csv \
        --out dataset --samples-per-vial 100 --seed 7
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

from render_geometry import STREAM_SURFACE_ALIGNMENT_MODEL


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
    "t200_ml": (0.45, 0.20, 0.0),
    "t400_ml": (0.0, 0.29, 0.45),
}
COLOR_MODEL = "beer_lambert_rgb_v6_video5_r05_r06_opacity_075"
TRANSMITTANCE_FLOOR_SRGB = 0.01
DEFAULT_LIQUID_HEIGHT_SCALE = 0.25
DEFAULT_LIQUID_OPACITY = 0.75
# Recipe 7 includes clear T100; lift its calibrated result toward neutral so
# the mixture reads as pale blue instead of the saturated T400 source color.
RECIPE_NEUTRAL_LIFT = {7: 0.10}
VIDEO5_PHASE_LINEAR_FACTORS = {
    5: ((0.1732, 1.1016, 0.7241), (0.0232, 0.2217, 0.0554)),
    6: ((0.4424, 1.1940, 0.6400), (0.4299, 1.1672, 0.7681)),
    7: ((0.600, 0.380, 0.600), (0.600, 0.430, 0.650)),
    8: ((0.120, 0.231, 0.509), (0.145, 0.225, 0.293)),
    9: ((0.025, 0.167, 0.361), (0.118, 0.151, 0.283)),
    10: ((3.011, 0.777, 0.315), (2.247, 0.601, 0.206)),
    11: ((0.908, 1.030, 1.851), (0.129, 0.215, 0.225)),
    12: ((0.226, 0.254, 0.136), (0.359, 0.290, 0.143)),
    13: ((0.062, 0.213, 0.183), (0.107, 0.226, 0.215)),
    14: ((68.424, 2.201, 0.490), (29.295, 1.325, 0.314)),
}


def srgb_to_linear(channel):
    return (
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
    )


def linear_to_srgb(channel):
    return (
        12.92 * channel
        if channel <= 0.0031308
        else 1.055 * channel ** (1 / 2.4) - 0.055
    )


def blend_color_srgb(primary, secondary, secondary_fraction):
    """Blend transparent colors by volume using channel-wise optical density."""
    if not 0 <= secondary_fraction <= 1:
        raise ValueError("secondary_fraction must be between 0 and 1")
    if secondary_fraction == 0:
        return list(primary)
    if secondary_fraction == 1:
        return list(secondary)
    result = []
    for primary_channel, secondary_channel in zip(primary, secondary):
        primary_linear = srgb_to_linear(
            max(primary_channel, TRANSMITTANCE_FLOOR_SRGB)
        )
        secondary_linear = srgb_to_linear(
            max(secondary_channel, TRANSMITTANCE_FLOOR_SRGB)
        )
        optical_density = (
            (1 - secondary_fraction) * -math.log(primary_linear)
            + secondary_fraction * -math.log(secondary_linear)
        )
        result.append(linear_to_srgb(math.exp(-optical_density)))
    return result


def recipe_color_srgb(tank_volumes_ml, total_ml):
    """Estimate the dyed phase color; clear T100 does not dilute that phase."""
    if total_ml <= 0:
        raise ValueError("total_ml must be positive")
    colored_volumes = {
        tank: max(0.0, float(tank_volumes_ml.get(tank, 0.0) or 0.0))
        for tank in ("t200_ml", "t400_ml")
    }
    if sum(colored_volumes.values()) > total_ml + 1e-6:
        raise ValueError("colored tank volume cannot exceed total_ml")
    active_tanks = [tank for tank, volume in colored_volumes.items() if volume > 0]
    if not active_tanks:
        return [1.0, 1.0, 1.0]
    colored_total_ml = sum(colored_volumes.values())
    if (
        len(active_tanks) == 1
        and math.isclose(
            colored_volumes[active_tanks[0]], colored_total_ml, abs_tol=1e-6
        )
    ):
        return list(TANK_COLORS_SRGB[active_tanks[0]])
    result = []
    for channel in range(3):
        optical_density = sum(
            (volume / colored_total_ml)
            * -math.log(
                srgb_to_linear(max(
                    TANK_COLORS_SRGB[tank][channel],
                    TRANSMITTANCE_FLOOR_SRGB,
                ))
            )
            for tank, volume in colored_volumes.items()
        )
        result.append(linear_to_srgb(math.exp(-optical_density)))
    return result


def _tank_field(token):
    tank = str(token).strip().lower().removeprefix("t")
    if tank.endswith("_ml"):
        tank = tank[:-3]
    if tank not in {"100", "200", "400"}:
        raise ValueError(f"unknown dispense tank: {token}")
    return f"t{tank}_ml"


def partial_recipe_volumes(tank_volumes_ml, dispense_order, dispensed_ml):
    """Return the recipe volumes dispensed so far, in the declared tank order."""
    volumes = {
        f"t{tank}_ml": max(
            0.0,
            float(
                tank_volumes_ml.get(
                    f"t{tank}_ml",
                    tank_volumes_ml.get(tank, 0.0),
                )
                or 0.0
            ),
        )
        for tank in (100, 200, 400)
    }
    remaining = max(0.0, float(dispensed_ml))
    partial = {field: 0.0 for field in volumes}
    for token in dispense_order:
        field = _tank_field(token)
        amount = min(volumes[field], remaining)
        partial[field] = amount
        remaining -= amount
        if remaining <= 1e-9:
            break
    return partial


def last_colored_tank(tank_volumes_ml, dispense_order):
    """Return the last non-clear tank that contributes liquid to a recipe."""
    volumes = {
        f"t{tank}_ml": float(
            tank_volumes_ml.get(
                f"t{tank}_ml",
                tank_volumes_ml.get(tank, 0.0),
            )
            or 0.0
        )
        for tank in (100, 200, 400)
    }
    for token in reversed(dispense_order):
        field = _tank_field(token)
        if field != "t100_ml" and volumes[field] > 0:
            return field
    return None


def recipe_color_at_progress_srgb(
    tank_volumes_ml,
    dispense_order,
    total_ml,
    dispensed_ml,
    carryover_ml=0.0,
    carryover_tank=None,
):
    """Estimate the liquid's color from ingredients dispensed up to this frame."""
    if total_ml <= 0:
        raise ValueError("total_ml must be positive")
    progress = min(float(total_ml), max(0.0, float(dispensed_ml)))
    if progress <= 1e-9:
        return [1.0, 1.0, 1.0]

    carryover = min(progress, max(0.0, float(carryover_ml)))
    recipe_progress = progress - carryover
    volumes = partial_recipe_volumes(
        tank_volumes_ml, dispense_order, recipe_progress
    )
    if carryover_tank is not None and carryover > 0:
        field = _tank_field(carryover_tank)
        volumes[field] += carryover
    return recipe_color_srgb(volumes, progress)


def calibrate_recipe_color_srgb(recipe_id, color, fill_fraction):
    """Apply video-5 ROI corrections measured at 50% and 75% fill."""
    phase_factors = VIDEO5_PHASE_LINEAR_FACTORS.get(int(recipe_id))
    if phase_factors is None:
        return list(color)
    fraction = min(1.0, max(0.0, float(fill_fraction)))
    points = (
        (0.0, (1.0, 1.0, 1.0)),
        (0.5, phase_factors[0]),
        (0.75, phase_factors[1]),
    )
    if fraction >= points[-1][0]:
        factors = points[-1][1]
    else:
        left, right = next(
            (left, right)
            for left, right in zip(points, points[1:])
            if left[0] <= fraction <= right[0]
        )
        phase = (fraction - left[0]) / (right[0] - left[0])
        factors = [
            left[1][channel] * (1.0 - phase) + right[1][channel] * phase
            for channel in range(3)
        ]
    calibrated = [
        linear_to_srgb(
            min(1.0, max(0.0, srgb_to_linear(channel) * factor))
        )
        for channel, factor in zip(color, factors)
    ]
    neutral_lift = RECIPE_NEUTRAL_LIFT.get(int(recipe_id), 0.0)
    if neutral_lift:
        calibrated = [
            linear_to_srgb(
                (1.0 - neutral_lift) * srgb_to_linear(channel) + neutral_lift
            )
            for channel in calibrated
        ]
    return calibrated


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
        metadata["liquid_color_srgb"] = recipe_color_srgb(metadata, volume)
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
            fill_fraction = (
                (frame - ANIMATION_START) / (ANIMATION_END - ANIMATION_START)
            )
            examples.append({
                **vial,
                "frame": frame,
                "liquid_color_srgb": calibrate_recipe_color_srgb(
                    vial["recipe_id"], vial["liquid_color_srgb"], fill_fraction
                ),
                "label": "pass",
                "defect": "",
                "proxy": False,
                "label_quality": "recipe_color_approximation",
                "color_model": COLOR_MODEL,
                "liquid_height_scale": DEFAULT_LIQUID_HEIGHT_SCALE,
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
                "--liquid-height-scale", str(DEFAULT_LIQUID_HEIGHT_SCALE),
                "--liquid-color", *(str(channel) for channel in example["liquid_color_srgb"]),
            ]
            record = {
                **example,
                "image": str(output.relative_to(out)),
                "camera": args.camera,
                "stage": str(stage),
                "seed": args.seed,
                "liquid_height_scale": DEFAULT_LIQUID_HEIGHT_SCALE,
                "stream_surface_alignment": (
                    STREAM_SURFACE_ALIGNMENT_MODEL if args.camera == "FillCam" else None
                ),
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
