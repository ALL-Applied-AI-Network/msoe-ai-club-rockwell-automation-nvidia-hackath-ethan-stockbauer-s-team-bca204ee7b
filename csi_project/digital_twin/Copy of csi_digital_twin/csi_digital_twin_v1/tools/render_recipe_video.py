"""Render a continuous two-vial recipe-configurable MP4 and frame-label CSV.

The recipe JSON format is the format used by ``video5_vial_recipes.json``:
each item contains ``vial``, ``recipe_id``, ``tanks``, ``total_ml``,
``dispense_order`` and optional descriptive fields.  Each vial is rendered as
one configurable-duration fill followed by an optional between-fill gap.

The input JSON must contain exactly two recipe objects.  In ``accepted`` mode
each vial is rendered with its intended recipe.  In ``rejected`` mode vial 1
is rendered as intended and vial 2 includes the last colored tank dispensed
from vial 1 as nozzle carryover; labels still identify vial 2 by its intended
recipe. Liquid composition progresses through the declared dispense order.

Example:
    <isaac-sim>/python.sh tools/render_recipe_video.py \
        --recipes data/two_vial_recipes.json \
        --out dataset/two_vial_accepted \
        --mode accepted --frames-per-vial 90 --gap-frames 6 --fps 30

The command writes ``<out>.mp4``, ``<out>_frame_labels.csv``,
``<out>_recipes.csv`` and ``<out>_metadata.json``. Rendering is performed
through render_sequence.py so the USD scene remains unchanged.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from render_batch1 import (
    ANIMATION_END,
    ANIMATION_START,
    COLOR_MODEL,
    DEFAULT_LIQUID_HEIGHT_SCALE,
    DEFAULT_ISAAC_PYTHON,
    HERE,
    RENDER,
    TANK_COLORS_SRGB,
    calibrate_recipe_color_srgb,
    last_colored_tank,
    positive_int,
    recipe_color_at_progress_srgb,
    recipe_color_srgb,
)
from render_geometry import STREAM_SURFACE_ALIGNMENT_MODEL

TANKS = ("100", "200", "400")
REFERENCE_FPS = 30
CSV_FIELDS = (
    "filename", "timestamp", "video_s", "in_fill", "vial", "recipe_id",
    "type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order",
    "color_purpose", )
RECIPE_FIELDS = (
    "vial", "recipe_id", "type", "t100_ml", "t200_ml", "t400_ml",
    "total_ml", "dispense_order", "color_purpose", "start_time", "end_time",
    "start_s", "end_s", "duration_s", "start_frame", "end_frame",
)


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def sampled_animation_frames(frames_per_vial, fps):
    animation_length = ANIMATION_END - ANIMATION_START + 1
    if frames_per_vial > animation_length:
        raise ValueError(f"frames-per-vial cannot exceed {animation_length}")
    if fps < 1 or fps > REFERENCE_FPS or REFERENCE_FPS % fps:
        raise ValueError("fps must be a positive divisor of 30, no greater than 30")
    output_count = max(1, round(frames_per_vial * fps / REFERENCE_FPS))
    if output_count == 1:
        return [ANIMATION_START]
    return [
        round(
            ANIMATION_START
            + index * (ANIMATION_END - ANIMATION_START) / (output_count - 1)
        )
        for index in range(output_count)
    ]


def read_recipes(path):
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list")

    if len(data) != 2:
        raise ValueError(f"{path} must contain exactly two recipes (vials 1 and 2)")

    recipes = []
    seen_vials = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each recipe must be a JSON object")
        missing = {"vial", "recipe_id", "tanks", "total_ml", "dispense_order"} - set(item)
        if missing:
            raise ValueError(f"recipe is missing fields: {', '.join(sorted(missing))}")
        vial = int(item["vial"])
        if vial in seen_vials or vial < 1:
            raise ValueError(f"invalid or duplicate vial: {vial}")
        seen_vials.add(vial)
        tanks = {tank: float(item["tanks"].get(tank, 0) or 0) for tank in TANKS}
        total_ml = float(item["total_ml"])
        if total_ml <= 0 or any(volume < 0 for volume in tanks.values()):
            raise ValueError(f"recipe {item['recipe_id']} has invalid volumes")
        if not math.isclose(sum(tanks.values()), total_ml, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"recipe {item['recipe_id']} total_ml does not equal its tank volumes"
            )
        order = tuple(
            token.strip().upper().removeprefix("T")
            for token in str(item["dispense_order"]).split(">")
            if token.strip()
        )
        active_tanks = {tank for tank, volume in tanks.items() if volume > 0}
        if (
            not order
            or any(tank not in TANKS for tank in order)
            or len(set(order)) != len(order)
            or set(order) != active_tanks
        ):
            raise ValueError(f"recipe {item['recipe_id']} has an invalid dispense_order")
        recipes.append({
            **item,
            "vial": vial,
            "recipe_id": int(item["recipe_id"]),
            "tanks": tanks,
            "total_ml": total_ml,
            "dispense_order": order,
        })
    recipes.sort(key=lambda recipe: recipe["vial"])
    if [recipe["vial"] for recipe in recipes] != [1, 2]:
        raise ValueError(f"{path} recipes must have vial values 1 and 2")
    return recipes


def weighted_color(recipe):
    return recipe_color_srgb(
        {f"t{tank}_ml": volume for tank, volume in recipe["tanks"].items()},
        recipe["total_ml"],
    )


def dispensing_color(recipe, render_frame, residual_ml=0.0, residual_recipe=None):
    progress = (
        max(0, min(ANIMATION_END - ANIMATION_START, render_frame - ANIMATION_START))
        / (ANIMATION_END - ANIMATION_START)
        * recipe["total_ml"]
    )
    if residual_recipe is not None and progress < residual_ml:
        residual_tank = last_colored_tank(
            {
                f"t{tank}_ml": volume
                for tank, volume in residual_recipe["tanks"].items()
            },
            residual_recipe["dispense_order"],
        )
        return (
            list(TANK_COLORS_SRGB[residual_tank])
            if residual_tank else [1.0, 1.0, 1.0]
        )
    progress = max(0.0, progress - residual_ml)
    accumulated = 0.0
    for tank in recipe["dispense_order"]:
        accumulated += recipe["tanks"][tank]
        if progress < accumulated:
            return list(TANK_COLORS_SRGB[f"t{tank}_ml"])
    return list(TANK_COLORS_SRGB[f"t{recipe['dispense_order'][-1]}_ml"])


def rendered_colors(recipe, render_frame, residual_recipe=None, residual_fraction=0.0):
    residual_ml = (
        recipe["total_ml"] * residual_fraction
        if residual_recipe is not None else 0.0
    )
    progress_ml = (
        max(0, min(ANIMATION_END - ANIMATION_START, render_frame - ANIMATION_START))
        / (ANIMATION_END - ANIMATION_START)
        * recipe["total_ml"]
    )
    carryover_tank = (
        last_colored_tank(
            {
                f"t{tank}_ml": volume
                for tank, volume in residual_recipe["tanks"].items()
            },
            residual_recipe["dispense_order"],
        )
        if residual_recipe is not None else None
    )
    liquid = recipe_color_at_progress_srgb(
        {f"t{tank}_ml": volume for tank, volume in recipe["tanks"].items()},
        recipe["dispense_order"],
        recipe["total_ml"],
        progress_ml,
        carryover_ml=residual_ml,
        carryover_tank=carryover_tank,
    )
    liquid = calibrate_recipe_color_srgb(
        recipe["recipe_id"], liquid, progress_ml / recipe["total_ml"]
    )
    stream = dispensing_color(
        recipe, render_frame, residual_ml, residual_recipe
    )
    return liquid, stream


def recipe_row(recipe, frame_index, fps, start_time, filename):
    timestamp = start_time + timedelta(seconds=frame_index / fps)
    return {
        "filename": filename,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "video_s": f"{frame_index / fps:.3f}",
        "in_fill": "1",
        "vial": recipe["vial"],
        "recipe_id": recipe["recipe_id"],
        "type": recipe.get("type", ""),
        "t100_ml": recipe["tanks"]["100"] or "",
        "t200_ml": recipe["tanks"]["200"] or "",
        "t400_ml": recipe["tanks"]["400"] or "",
        "total_ml": recipe["total_ml"],
        "dispense_order": ">".join(f"T{tank}" for tank in recipe["dispense_order"]),
        "color_purpose": recipe.get("color_purpose", ""),
    }


def gap_row(frame_index, fps, start_time, filename):
    timestamp = start_time + timedelta(seconds=frame_index / fps)
    return {
        "filename": filename,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "video_s": f"{frame_index / fps:.3f}",
        "in_fill": "0",
        "vial": "",
        "recipe_id": "",
        "type": "between_fills",
        "t100_ml": "",
        "t200_ml": "",
        "t400_ml": "",
        "total_ml": "",
        "dispense_order": "",
        "color_purpose": "",
    }


def build_timeline(
    recipes, render_frames, gap_frames, fps, start_time, mode, residual_fraction
):
    timeline = []
    frame_index = 0
    for recipe in recipes:
        for render_frame in render_frames:
            filename = f"frame_{frame_index:06d}.png"
            residual_recipe = (
                recipes[0]
                if mode == "rejected" and recipe["vial"] == 2
                else None
            )
            liquid_color, stream_color = rendered_colors(
                recipe, render_frame, residual_recipe, residual_fraction
            )
            timeline.append(
                (
                    recipe_row(recipe, frame_index, fps, start_time, filename),
                    recipe,
                    render_frame,
                    liquid_color,
                    stream_color,
                )
            )
            frame_index += 1
        for _ in range(gap_frames):
            filename = f"frame_{frame_index:06d}.png"
            timeline.append(
                (
                    gap_row(frame_index, fps, start_time, filename),
                    None,
                    ANIMATION_START,
                    None,
                    None,
                )
            )
            frame_index += 1
    return timeline


def build_recipe_rows(recipes, frames_per_vial, gap_frames, fps, start_time):
    rows = []
    frame_index = 0
    for recipe in recipes:
        end_index = frame_index + frames_per_vial - 1
        start_timestamp = start_time + timedelta(seconds=frame_index / fps)
        end_timestamp = start_time + timedelta(seconds=end_index / fps)
        rows.append({
            "vial": recipe["vial"],
            "recipe_id": recipe["recipe_id"],
            "type": recipe.get("type", ""),
            "t100_ml": recipe["tanks"]["100"] or "",
            "t200_ml": recipe["tanks"]["200"] or "",
            "t400_ml": recipe["tanks"]["400"] or "",
            "total_ml": recipe["total_ml"],
            "dispense_order": ">".join(f"T{tank}" for tank in recipe["dispense_order"]),
            "color_purpose": recipe.get("color_purpose", ""),
            "start_time": start_timestamp.strftime("%H:%M:%S.%f")[:-3],
            "end_time": end_timestamp.strftime("%H:%M:%S.%f")[:-3],
            "start_s": f"{frame_index / fps:.3f}",
            "end_s": f"{end_index / fps:.3f}",
            "duration_s": f"{(end_index - frame_index) / fps:.3f}",
            "start_frame": frame_index,
            "end_frame": end_index,
        })
        frame_index += frames_per_vial + gap_frames
    return rows


def resolve_isaac_python(value, dry_run):
    configured = value or os.environ.get("ISAAC_SIM_PYTHON")
    path = Path(configured).expanduser().resolve() if configured else DEFAULT_ISAAC_PYTHON
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"Isaac Sim Python not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True, help="recipe JSON file")
    parser.add_argument("--out", type=Path, required=True,
                        help="output prefix, without .mp4 or _frame_labels.csv")
    parser.add_argument("--camera", choices=("FillCam", "CapSealCam"), default="FillCam")
    parser.add_argument("--mode", choices=("accepted", "rejected"), default="accepted")
    parser.add_argument(
        "--residual-fraction", type=float, default=0.10,
        help="rejected-mode fraction of vial 1 color carried into vial 2 (0..1)",
    )
    parser.add_argument("--stage", type=Path, default=HERE / "csi_line.usda")
    parser.add_argument(
        "--frames-per-vial", type=positive_int,
        default=ANIMATION_END - ANIMATION_START + 1,
        help="output frames per vial; use 686 for every animation frame",
    )
    parser.add_argument("--gap-frames", type=nonnegative_int, default=0)
    parser.add_argument(
        "--fps", type=positive_int, default=30,
        help="output FPS; lower divisors of 30 sample fewer animation frames",
    )
    parser.add_argument("--subframes", type=positive_int, default=8)
    parser.add_argument("--start-timestamp", default="2026-01-01T00:00:00")
    parser.add_argument("--isaac-python", type=Path, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.residual_fraction <= 1:
        parser.error("--residual-fraction must be between 0 and 1")
    try:
        render_frames = sampled_animation_frames(args.frames_per_vial, args.fps)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "accepted" and args.residual_fraction != 0.10:
        parser.error("--residual-fraction is only used with --mode rejected")

    recipes = read_recipes(args.recipes.expanduser().resolve())
    stage = args.stage.expanduser().resolve()
    if not stage.is_file():
        parser.error(f"stage does not exist: {stage}")
    try:
        start_time = datetime.fromisoformat(args.start_timestamp)
    except ValueError as error:
        parser.error(f"--start-timestamp must be ISO-8601: {error}")
    isaac_python = resolve_isaac_python(args.isaac_python, args.dry_run)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out.parent / f".{out.name}_frames"
    video_path = out.with_suffix(".mp4")
    labels_path = out.parent / f"{out.name}_frame_labels.csv"
    recipes_path = out.parent / f"{out.name}_recipes.csv"
    metadata_path = out.parent / f"{out.name}_metadata.json"
    plan_path = out.parent / f".{out.name}_render_plan.json"
    output_gap_frames = round(args.gap_frames * args.fps / REFERENCE_FPS)
    output_frames_per_vial = len(render_frames)
    timeline = build_timeline(
        recipes, render_frames, output_gap_frames, args.fps, start_time,
        args.mode, args.residual_fraction,
    )
    recipe_rows = build_recipe_rows(
        recipes, output_frames_per_vial, output_gap_frames, args.fps, start_time
    )
    if not args.dry_run and shutil.which(args.ffmpeg) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg}")
    if frame_dir.exists() and not args.dry_run:
        raise FileExistsError(f"temporary frame directory already exists: {frame_dir}")
    if not args.dry_run:
        frame_dir.mkdir()
    try:
        with recipes_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECIPE_FIELDS)
            writer.writeheader()
            writer.writerows(recipe_rows)
        with labels_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row, recipe, render_frame, liquid_color, stream_color in timeline:
                writer.writerow(row)
        if not args.dry_run:
            plan_path.write_text(
                json.dumps(
                    [
                        {
                            "output": str(frame_dir / row["filename"]),
                            "render_frame": render_frame,
                            "liquid_color": liquid_color,
                            "stream_color": stream_color,
                        }
                        for row, recipe, render_frame, liquid_color, stream_color in timeline
                    ]
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    str(isaac_python), str(HERE / "tools" / "render_sequence.py"),
                    "--plan", str(plan_path), "--camera", args.camera,
                    "--stage", str(stage), "--subframes", str(args.subframes),
                    "--liquid-height-scale", str(DEFAULT_LIQUID_HEIGHT_SCALE),
                ],
                check=True,
                cwd=str(HERE),
            )
            plan_path.unlink()
        if not args.dry_run:
            subprocess.run([
                args.ffmpeg, "-y", "-framerate", str(args.fps),
                "-i", str(frame_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-movflags", "+faststart", str(video_path),
            ], check=True, cwd=str(out.parent))
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": args.mode,
                    "vials": [recipe["vial"] for recipe in recipes],
                    "residual_source_vial": 1 if args.mode == "rejected" else None,
                    "residual_target_vial": 2 if args.mode == "rejected" else None,
                    "residual_fraction": (
                        args.residual_fraction if args.mode == "rejected" else 0.0
                    ),
                    "frames_per_vial": args.frames_per_vial,
                    "gap_frames": args.gap_frames,
                    "fps": args.fps,
                    "color_model": COLOR_MODEL,
                    "liquid_height_scale": DEFAULT_LIQUID_HEIGHT_SCALE,
                    "stream_surface_alignment": STREAM_SURFACE_ALIGNMENT_MODEL,
                },
                handle,
                indent=2,
            )
    finally:
        if frame_dir.exists() and not args.keep_frames:
            shutil.rmtree(frame_dir)
        if plan_path.exists():
            plan_path.unlink()
    mode = "planned" if args.dry_run else "rendered"
    print(f"{mode} {len(timeline)} frames")
    print("video:", video_path)
    print("labels:", labels_path)
    print("recipes:", recipes_path)
    print("metadata:", metadata_path)


if __name__ == "__main__":
    main()
