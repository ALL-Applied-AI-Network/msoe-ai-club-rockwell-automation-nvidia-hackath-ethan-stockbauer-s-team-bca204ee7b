"""Generate ordered two-recipe accepted and contamination video datapoints.

Each datapoint is a continuous FillCam clip: recipe A fills vial 1, then an
empty vial 2 arrives and follows recipe B. Ordered pairs are used because
carryover is directional. By default one rejected datapoint is made per pair,
cycling contamination volumes from 2 through 10 ml; pass datapoints are
uncontaminated. Lower FPS values sample animation frames at the corresponding
30 FPS interval while preserving the clip's approximate duration.
"""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

from render_batch1 import (
    ANIMATION_END,
    ANIMATION_START,
    COLOR_MODEL,
    DEFAULT_LIQUID_HEIGHT_SCALE,
    DEFAULT_LIQUID_OPACITY,
    TANK_COLORS_SRGB,
    calibrate_recipe_color_srgb,
    last_colored_tank,
    recipe_color_at_progress_srgb,
    recipe_color_srgb,
)
from render_geometry import STREAM_SURFACE_ALIGNMENT_MODEL

REFERENCE_FPS = 30
LABEL_FIELDS = (
    "filename", "timestamp", "video_s", "in_fill", "vial", "recipe_id",
    "type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order",
    "color_purpose", "mp4_frame", "render_frame", "pair_id", "scenario",
    "contamination_ml", "contamination_tank", "recipe_before", "recipe_after",
    "label",
)


def parse_recipes(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"recipe_id", "type", "t100_ml", "t200_ml", "t400_ml", "total_ml",
                "dispense_order", "color_purpose"}
    missing = required.difference(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    recipes = {}
    for row in rows:
        recipe_id = int(row["recipe_id"])
        volumes = {
            tank: float(row[f"t{tank}_ml"]) if row[f"t{tank}_ml"] else 0.0
            for tank in (100, 200, 400)
        }
        total = float(row["total_ml"])
        if total <= 0 or not math.isclose(sum(volumes.values()), total, abs_tol=1e-6):
            raise ValueError(f"invalid volumes for recipe {recipe_id}")
        order = tuple(token.strip().upper() for token in row["dispense_order"].split(">"))
        ordered_tanks = tuple(
            int(token.removeprefix("T")) for token in order
            if token.startswith("T") and token[1:].isdigit()
        )
        active_tanks = {tank for tank, volume in volumes.items() if volume > 0}
        if (
            len(ordered_tanks) != len(order)
            or len(set(ordered_tanks)) != len(ordered_tanks)
            or set(ordered_tanks) != active_tanks
        ):
            raise ValueError(
                f"recipe {recipe_id} dispense_order must list each non-empty tank once"
            )
        if recipe_id in recipes and recipes[recipe_id]["volumes"] != volumes:
            raise ValueError(f"recipe {recipe_id} has inconsistent rows")
        recipes[recipe_id] = {
            "recipe_id": recipe_id,
            "type": row["type"],
            "volumes": volumes,
            "total_ml": total,
            "order": order,
            "purpose": row["color_purpose"],
            "color": recipe_color_srgb(
                {f"t{tank}_ml": volume for tank, volume in volumes.items()},
                total,
            ),
        }
    return [recipes[key] for key in sorted(recipes)]


def parse_recipe_pair(value):
    try:
        before, after = (int(part) for part in value.split(":", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("recipe pair must use BEFORE:AFTER") from error
    if before <= 0 or after <= 0 or before == after:
        raise argparse.ArgumentTypeError("recipe pair IDs must be positive and distinct")
    return before, after


def tank_color(tank):
    key = str(tank).strip().lower()
    if not key.startswith("t"):
        key = f"t{key}"
    if not key.endswith("_ml"):
        key = f"{key}_ml"
    return list(TANK_COLORS_SRGB[key])


def sampled_render_frames(fps):
    if fps < 1 or fps > REFERENCE_FPS or REFERENCE_FPS % fps:
        raise ValueError("fps must be a positive divisor of 30, no greater than 30")
    step = REFERENCE_FPS // fps
    frames = list(range(ANIMATION_START, ANIMATION_END + 1, step))
    if frames[-1] != ANIMATION_END:
        frames[-1] = ANIMATION_END
    return frames


def frame_progress_ml(recipe, frame):
    fraction = (
        max(0, min(ANIMATION_END - ANIMATION_START, frame - ANIMATION_START))
        / max(1, ANIMATION_END - ANIMATION_START)
    )
    return fraction * recipe["total_ml"]


def stream_color(recipe, frame, residual_ml=0.0, previous=None):
    progress_ml = frame_progress_ml(recipe, frame)
    if previous is not None and progress_ml < residual_ml:
        residual_tank = last_colored_tank(
            {f"t{tank}_ml": volume for tank, volume in previous["volumes"].items()},
            previous["order"],
        )
        return tank_color(residual_tank) if residual_tank else [1.0, 1.0, 1.0]
    progress_ml = max(0.0, progress_ml - residual_ml)
    accumulated = 0.0
    for token in recipe["order"]:
        tank = int(token.removeprefix("T"))
        accumulated += recipe["volumes"][tank]
        if progress_ml < accumulated:
            return tank_color(tank)
    return tank_color(recipe["order"][-1])


def build_plan(recipes, out, scenario, contamination_values, fps, recipe_pairs=None):
    plan = []
    labels = []
    datapoints = []
    render_frames = sampled_render_frames(fps)
    recipes_by_id = {recipe["recipe_id"]: recipe for recipe in recipes}
    pairs = recipe_pairs or [
        (before["recipe_id"], after["recipe_id"])
        for before in recipes
        for after in recipes
        if before["recipe_id"] != after["recipe_id"]
    ]
    for before_id, after_id in pairs:
        if before_id not in recipes_by_id or after_id not in recipes_by_id:
            raise ValueError(f"unknown recipe pair: {before_id}:{after_id}")
        before = recipes_by_id[before_id]
        after = recipes_by_id[after_id]
        pair_id = f"r{before['recipe_id']:02d}_to_r{after['recipe_id']:02d}"
        values = (0.0,) if scenario == "accepted" else contamination_values
        for contamination_ml in values:
            stem = f"{pair_id}_{'pass' if not contamination_ml else f'fail_{int(contamination_ml):02d}ml'}"
            video_path = out / "videos" / f"{stem}.mp4"
            label_path = out / "labels" / f"{stem}.csv"
            frame_dir = out / "frames" / stem
            scenario_name = "accepted" if not contamination_ml else "rejected"
            previous_tank = last_colored_tank(
                {
                    f"t{tank}_ml": volume
                    for tank, volume in before["volumes"].items()
                },
                before["order"],
            )
            video_frame = 0
            for vial, recipe, residual_ml, residual_tank in (
                (1, before, 0.0, None),
                (2, after, contamination_ml, previous_tank),
            ):
                for render_frame in render_frames:
                    progress_ml = frame_progress_ml(recipe, render_frame)
                    output = frame_dir / f"frame_{video_frame:06d}.png"
                    labels.append({
                        "filename": output.name,
                        "timestamp": f"{video_frame / fps:.6f}",
                        "video_s": f"{video_frame / fps:.6f}",
                        "in_fill": 1,
                        "vial": vial,
                        "recipe_id": recipe["recipe_id"],
                        "type": recipe["type"],
                        "t100_ml": recipe["volumes"][100] or "",
                        "t200_ml": recipe["volumes"][200] or "",
                        "t400_ml": recipe["volumes"][400] or "",
                        "total_ml": recipe["total_ml"],
                        "dispense_order": ">".join(recipe["order"]),
                        "color_purpose": recipe["purpose"],
                        "mp4_frame": video_frame,
                        "render_frame": render_frame,
                        "pair_id": pair_id,
                        "scenario": scenario_name,
                        "contamination_ml": contamination_ml,
                        "contamination_tank": (
                            f"T{residual_tank[1:4]}"
                            if vial == 2 and residual_tank else ""
                        ),
                        "recipe_before": before["recipe_id"],
                        "recipe_after": after["recipe_id"],
                        "label": "pass" if not contamination_ml else "reject",
                    })
                    plan.append({
                        "output": str(output),
                        "video_frame": video_frame,
                        "render_frame": render_frame,
                        "liquid_color": calibrate_recipe_color_srgb(
                            recipe["recipe_id"],
                            recipe_color_at_progress_srgb(
                                {
                                    f"t{tank}_ml": volume
                                    for tank, volume in recipe["volumes"].items()
                                },
                                recipe["order"],
                                recipe["total_ml"],
                                progress_ml,
                                carryover_ml=residual_ml,
                                carryover_tank=(
                                    residual_tank if vial == 2 else None
                                ),
                            ),
                            progress_ml / recipe["total_ml"],
                        ),
                        "stream_color": (
                            stream_color(
                                recipe, render_frame, residual_ml, before
                            )
                            if vial == 2
                            else stream_color(recipe, render_frame)
                        ),
                    })
                    video_frame += 1
            datapoints.append({
                "pair_id": pair_id,
                "scenario": scenario_name,
                "contamination_ml": contamination_ml,
                "recipe_before": before["recipe_id"],
                "recipe_after": after["recipe_id"],
                "video": str(video_path.relative_to(out)),
                "labels": str(label_path.relative_to(out)),
                "frames": len(render_frames) * 2,
                "fps": fps,
            })
    return plan, labels, datapoints


def encode_datapoints(datapoints, out, fps):
    for datapoint in datapoints:
        frame_dir = out / "frames" / Path(datapoint["video"]).stem
        video_path = out / datapoint["video"]
        video_path.parent.mkdir(parents=True, exist_ok=True)
        expected = int(datapoint["frames"])
        actual = len(list(frame_dir.glob("frame_*.png")))
        if actual != expected:
            raise RuntimeError(
                f"{frame_dir}: expected {expected} frames, found {actual}"
            )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps), "-i", str(frame_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            check=True,
        )
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "v:0", "-show_entries",
                "stream=nb_read_frames", "-of", "json", str(video_path),
            ],
            check=True, capture_output=True, text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_read_frames"]) != expected:
            raise RuntimeError(
                f"{video_path}: encoded frame count does not match labels"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=("accepted", "rejected", "both"), default="both")
    parser.add_argument("--contamination-ml", default="2,3,4,5,6,7,8,9,10")
    parser.add_argument(
        "--recipe-pair", action="append", type=parse_recipe_pair,
        help="render a specific ordered pair (repeatable), e.g. --recipe-pair 5:6",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--subframes", type=int, default=1)
    parser.add_argument("--isaac-python", type=Path, default=Path("/isaac-sim/python.sh"))
    parser.add_argument(
        "--stage", type=Path,
        default=Path(__file__).resolve().parents[1] / "stations/fill_station/fill_station.usda",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        sampled_render_frames(args.fps)
    except ValueError as error:
        parser.error(str(error))
    contamination = tuple(float(value) for value in args.contamination_ml.split(","))
    if not contamination or any(value < 2 or value > 10 for value in contamination):
        parser.error("--contamination-ml values must be between 2 and 10")
    recipes = parse_recipes(args.labels_csv)
    scenarios = ("accepted", "rejected") if args.scenario == "both" else (args.scenario,)
    args.out.mkdir(parents=True, exist_ok=True)
    all_datapoints = []
    for scenario in scenarios:
        plan, labels, datapoints = build_plan(
            recipes, args.out / scenario, scenario, contamination, args.fps,
            args.recipe_pair,
        )
        scenario_out = args.out / scenario
        scenario_out.mkdir(parents=True, exist_ok=True)
        (scenario_out / "render_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with (scenario_out / "labels_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            writer.writerows(labels)
        grouped = {}
        for row in labels:
            key = (row["pair_id"], row["contamination_ml"])
            grouped.setdefault(key, []).append(row)
        for rows in grouped.values():
            label_path = scenario_out / "labels" / (
                f"{rows[0]['pair_id']}_{'pass' if not rows[0]['contamination_ml'] else f'fail_{int(float(rows[0]['contamination_ml'])):02d}ml'}.csv"
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        if not args.dry_run:
            subprocess.run(
                [
                    str(args.isaac_python), str(Path(__file__).with_name("render_sequence.py")),
                    "--plan", str(scenario_out / "render_plan.json"),
                    "--stage", str(args.stage),
                    "--subframes", str(args.subframes),
                    "--liquid-height-scale", str(DEFAULT_LIQUID_HEIGHT_SCALE),
                ],
                check=True,
            )
            encode_datapoints(datapoints, scenario_out, args.fps)
        for datapoint in datapoints:
            datapoint["video"] = str(Path(scenario) / datapoint["video"])
            datapoint["labels"] = str(Path(scenario) / datapoint["labels"])
        all_datapoints.extend(datapoints)
    with (args.out / "datapoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_datapoints[0])
        writer.writeheader()
        writer.writerows(all_datapoints)
    (args.out / "dataset_metadata.json").write_text(json.dumps({
        "ordered_pairs": True,
        "recipe_ids": [recipe["recipe_id"] for recipe in recipes],
        "recipe_pairs": args.recipe_pair or "all ordered pairs",
        "scenarios": scenarios,
        "contamination_ml": contamination,
        "frames_per_vial": len(sampled_render_frames(args.fps)),
        "animation_sampling_step": REFERENCE_FPS // args.fps,
        "fps": args.fps,
        "liquid_opacity": DEFAULT_LIQUID_OPACITY,
        "liquid_height_scale": DEFAULT_LIQUID_HEIGHT_SCALE,
        "stream_opacity": 0.42,
        "liquid_color_flow_gated": True,
        "color_model": COLOR_MODEL,
        "stream_surface_alignment": STREAM_SURFACE_ALIGNMENT_MODEL,
    }, indent=2), encoding="utf-8")
    print(f"planned {len(all_datapoints)} datapoints")


if __name__ == "__main__":
    main()
