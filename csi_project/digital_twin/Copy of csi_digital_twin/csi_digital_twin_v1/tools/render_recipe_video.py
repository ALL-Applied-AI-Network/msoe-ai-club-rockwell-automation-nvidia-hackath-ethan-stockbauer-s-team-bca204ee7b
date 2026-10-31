"""Render a recipe-configurable MP4 and frame-label CSV.

The recipe JSON format is the format used by ``video5_vial_recipes.json``:
each item contains ``vial``, ``recipe_id``, ``tanks``, ``total_ml``,
``dispense_order`` and optional descriptive fields.  Each vial is rendered as
one configurable-duration fill followed by an optional between-fill gap.

Example:
    <isaac-sim>/python.sh tools/render_recipe_video.py \
        --recipes data/video5_vial_recipes.json \
        --out dataset/video5_synthetic \
        --frames-per-vial 90 --gap-frames 6 --fps 30

The command writes ``<out>.mp4``, ``<out>_frame_labels.csv`` and
``<out>_recipes.csv``. Rendering is performed through render_example.py so the
USD scene remains unchanged.
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
    DEFAULT_ISAAC_PYTHON,
    HERE,
    RENDER,
    TANK_COLORS_SRGB,
    positive_int,
)

TANKS = ("100", "200", "400")
CSV_FIELDS = (
    "filename", "timestamp", "video_s", "in_fill", "vial", "recipe_id",
    "type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order",
    "color_purpose", "mp4_frame", "render_frame",
)
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


def read_recipes(path):
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list")

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
        if not order or any(tank not in TANKS for tank in order):
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
    return recipes


def weighted_color(recipe):
    return [
        sum(
            recipe["tanks"][tank] * TANK_COLORS_SRGB[f"t{tank}_ml"][channel]
            for tank in TANKS
        ) / recipe["total_ml"]
        for channel in range(3)
    ]


def dispensing_color(recipe, render_frame):
    progress = (
        max(0, min(ANIMATION_END - ANIMATION_START, render_frame - ANIMATION_START))
        / (ANIMATION_END - ANIMATION_START)
        * recipe["total_ml"]
    )
    accumulated = 0.0
    for tank in recipe["dispense_order"]:
        accumulated += recipe["tanks"][tank]
        if progress < accumulated:
            return list(TANK_COLORS_SRGB[f"t{tank}_ml"])
    return weighted_color(recipe)


def recipe_row(recipe, frame_index, render_frame, fps, start_time, filename):
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
        "mp4_frame": frame_index,
        "render_frame": render_frame,
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
        "mp4_frame": frame_index,
        "render_frame": ANIMATION_START,
    }


def build_timeline(recipes, frames_per_vial, gap_frames, fps, start_time):
    timeline = []
    frame_index = 0
    for recipe in recipes:
        for position in range(frames_per_vial):
            render_frame = round(
                ANIMATION_START
                + position * (ANIMATION_END - ANIMATION_START)
                / max(1, frames_per_vial - 1)
            )
            filename = f"frame_{frame_index:06d}.png"
            timeline.append((
                recipe_row(recipe, frame_index, render_frame, fps, start_time, filename),
                recipe,
                render_frame,
            ))
            frame_index += 1
        for _ in range(gap_frames):
            filename = f"frame_{frame_index:06d}.png"
            timeline.append((
                gap_row(frame_index, fps, start_time, filename),
                None,
                ANIMATION_START,
            ))
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
    parser.add_argument("--stage", type=Path, default=HERE / "csi_line.usda")
    parser.add_argument("--frames-per-vial", type=positive_int, default=90)
    parser.add_argument("--gap-frames", type=nonnegative_int, default=6)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--subframes", type=positive_int, default=8)
    parser.add_argument("--start-timestamp", default="2026-01-01T00:00:00")
    parser.add_argument("--isaac-python", type=Path, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
    timeline = build_timeline(
        recipes, args.frames_per_vial, args.gap_frames, args.fps, start_time
    )
    recipe_rows = build_recipe_rows(
        recipes, args.frames_per_vial, args.gap_frames, args.fps, start_time
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
            for row, recipe, render_frame in timeline:
                writer.writerow(row)
                if args.dry_run:
                    continue
                output = frame_dir / row["filename"]
                command = [
                    str(isaac_python), str(RENDER), "--camera", args.camera,
                    "--stage", str(stage), "--frame", str(render_frame),
                    "--subframes", str(args.subframes), "--out", str(output),
                ]
                if recipe is not None:
                    command.extend([
                        "--liquid-color", *(str(channel) for channel in weighted_color(recipe)),
                        "--stream-color", *(str(channel) for channel in dispensing_color(recipe, render_frame)),
                    ])
                subprocess.run(command, check=True, cwd=str(HERE))
        if not args.dry_run:
            subprocess.run([
                args.ffmpeg, "-y", "-framerate", str(args.fps),
                "-i", str(frame_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-movflags", "+faststart", str(video_path),
            ], check=True, cwd=str(out.parent))
    finally:
        if frame_dir.exists() and not args.keep_frames:
            shutil.rmtree(frame_dir)
    mode = "planned" if args.dry_run else "rendered"
    print(f"{mode} {len(timeline)} frames")
    print("video:", video_path)
    print("labels:", labels_path)
    print("recipes:", recipes_path)


if __name__ == "__main__":
    main()
