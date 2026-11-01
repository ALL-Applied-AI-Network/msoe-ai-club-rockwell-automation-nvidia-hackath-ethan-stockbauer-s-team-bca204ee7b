from __future__ import annotations

import argparse
import json
from pathlib import Path

from mixing_copilot.alignment.pipeline import run_alignment
from mixing_copilot.audit.manifest import audit_dataset
from mixing_copilot.config import load_yaml, resolve_path
from mixing_copilot.dataset.build import build_datasets, extract_features
from mixing_copilot.dataset.synthetic import process_synthetic_dataset
from mixing_copilot.extraction.frames import extract_vial_frames
from mixing_copilot.io.files import atomic_json_dump
from mixing_copilot.prediction import predict_vial, write_prediction


def _dataset_config(path: str) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    return config, resolve_path(config["dataset_root"], config_path.parent)


def command_audit(args: argparse.Namespace) -> int:
    config, root = _dataset_config(args.config)
    output = resolve_path(config.get("output_root", ".")) / "reports" / "dataset_audit.json"
    if args.dry_run:
        print(json.dumps({"dataset_root": str(root), "output": str(output)}, indent=2))
        return 0
    print(f"Auditing read-only dataset: {root}")
    report = audit_dataset(root, include_hashes=not args.skip_hashes)
    atomic_json_dump(report, output)
    print(f"Files: {report['file_count']}; missing required: {len(report['missing_required'])}")
    print(f"Wrote: {output}")
    return 0


def command_align(args: argparse.Namespace) -> int:
    config, root = _dataset_config(args.config)
    output_root = resolve_path(config.get("output_root", "."))
    camera = args.camera or config.get("camera_id", "vial_fill")
    video_id = args.video_id or int(config.get("video_id", 5))
    if args.dry_run:
        print(
            json.dumps(
                {"dataset_root": str(root), "camera": camera, "video_id": video_id}, indent=2
            )
        )
        return 0
    result = run_alignment(root, output_root, camera, video_id, config.get("alignment", {}))
    print(json.dumps(result["fit"], indent=2))
    print(f"Visual QA: {result['visual_qa']}")
    print(f"Manifest: {result['manifest']}")
    return 0


def command_extract(args: argparse.Namespace) -> int:
    config, root = _dataset_config(args.config)
    output_root = resolve_path(config.get("output_root", "."))
    camera = args.camera or config.get("camera_id", "vial_fill")
    video_id = args.video_id or int(config.get("video_id", 5))
    if args.dry_run:
        print(
            json.dumps(
                {"camera": camera, "video_id": video_id, "stages": config["extraction"]["stages"]},
                indent=2,
            )
        )
        return 0
    manifest = extract_vial_frames(
        root, output_root, camera, video_id, config.get("extraction", {})
    )
    count = manifest["vial_id"].nunique()
    print(f"Extracted {len(manifest)} supported stage frames for {count} vials")
    print(f"Manifest: {output_root / 'data' / 'interim' / 'video5_extracted_frames.csv'}")
    print(f"Contact sheets: {output_root / 'artifacts' / 'qa' / 'contact_sheets'}")
    return 0


def command_features(args: argparse.Namespace) -> int:
    output_root = Path(".").resolve()
    result = extract_features(output_root, Path(args.config).resolve())
    print(
        f"Measured {len(result)} frames; status counts: {result['status'].value_counts().to_dict()}"
    )
    print(f"QA overlays: {output_root / 'artifacts' / 'qa' / 'color'}")
    return 0


def command_build_dataset(args: argparse.Namespace) -> int:
    config, root = _dataset_config(args.config)
    frames, vials = build_datasets(root, resolve_path(config.get("output_root", ".")))
    print(f"Built {len(frames)} per-frame rows and {len(vials)} per-vial rows (CSV + Parquet)")
    return 0


def command_predict(args: argparse.Namespace) -> int:
    config, root = _dataset_config(args.config)
    output_root = resolve_path(config.get("output_root", "."))
    recipes = __import__("pandas").read_csv(
        root / "vial_fill" / "labels" / "video5_vial_recipes.csv"
    )
    recipe = recipes.loc[recipes["vial"] == args.vial_id].iloc[0].to_dict()
    manifest = __import__("pandas").read_csv(
        output_root / "data" / "interim" / "video5_extracted_frames.csv"
    )
    selected = manifest.loc[
        (manifest["vial_id"] == args.vial_id) & manifest["stage"].isin(["late_fill", "post_fill"])
    ]
    frames = [output_root / path for path in selected["image_path"]]
    previous = None
    if args.vial_id > 1:
        previous = recipes.loc[recipes["vial"] == args.vial_id - 1].iloc[0].to_dict()
    prediction = predict_vial(
        frames, recipe, previous, {"confidence": float(selected["alignment_confidence"].median())}
    )
    destination = output_root / "artifacts" / "predictions" / f"vial_{args.vial_id:02d}.json"
    write_prediction(prediction, destination)
    print(prediction.model_dump_json(indent=2))
    print(f"Wrote: {destination}")
    return 0


def command_process_synthetic(args: argparse.Namespace) -> int:
    result = process_synthetic_dataset(
        video_path=Path(args.video),
        frame_labels_path=Path(args.frame_labels),
        recipes_path=Path(args.recipes),
        output_root=Path(args.output_root),
        dataset_id=args.dataset_id,
        features_config_path=Path(args.features_config),
        camera_id=args.camera_id,
        video_id=args.video_id,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    common = argparse.Namespace(config=args.config, camera=None, video_id=5, dry_run=False)
    if not args.skip_audit:
        command_audit(
            argparse.Namespace(config=args.config, dry_run=False, skip_hashes=args.skip_hashes)
        )
    command_align(common)
    command_extract(common)
    command_features(argparse.Namespace(config="configs/features.yaml"))
    command_build_dataset(argparse.Namespace(config=args.config))
    command_predict(argparse.Namespace(config=args.config, vial_id=args.example_vial))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mixing-copilot", description=__doc__)
    parser.add_argument("--version", action="version", version="mixing-copilot 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="inventory and validate the external CSI dataset")
    audit.add_argument("--config", default="configs/dataset.yaml")
    audit.add_argument("--skip-hashes", action="store_true", help="faster metadata-only audit")
    audit.add_argument("--dry-run", action="store_true")
    audit.set_defaults(handler=command_audit)
    align = subparsers.add_parser("align", help="visually match stills and fit timestamp alignment")
    align.add_argument("--config", default="configs/dataset.yaml")
    align.add_argument("--camera")
    align.add_argument("--video-id", type=int)
    align.add_argument("--dry-run", action="store_true")
    align.set_defaults(handler=command_align)
    extract = subparsers.add_parser("extract", help="extract alignment-corrected vial stage frames")
    extract.add_argument("--config", default="configs/dataset.yaml")
    extract.add_argument("--camera")
    extract.add_argument("--video-id", type=int)
    extract.add_argument("--dry-run", action="store_true")
    extract.set_defaults(handler=command_extract)
    features = subparsers.add_parser(
        "features", help="measure liquid color and generate QA overlays"
    )
    features.add_argument("--config", default="configs/features.yaml")
    features.set_defaults(handler=command_features)
    build = subparsers.add_parser(
        "build-dataset", help="write ML-ready per-frame and per-vial tables"
    )
    build.add_argument("--config", default="configs/dataset.yaml")
    build.set_defaults(handler=command_build_dataset)
    predict = subparsers.add_parser("predict", help="write an example dashboard prediction")
    predict.add_argument("--config", default="configs/dataset.yaml")
    predict.add_argument("--vial-id", type=int, required=True)
    predict.set_defaults(handler=command_predict)
    synthetic = subparsers.add_parser(
        "process-synthetic",
        help="validate and process an Isaac Sim recipe video without CSI timestamp alignment",
    )
    synthetic.add_argument("--video", required=True, help="synthetic MP4 path")
    synthetic.add_argument("--frame-labels", required=True, help="*_frame_labels.csv path")
    synthetic.add_argument("--recipes", required=True, help="*_recipes.csv path")
    synthetic.add_argument("--dataset-id", required=True, help="safe identifier for output folders")
    synthetic.add_argument("--video-id", type=int, default=1)
    synthetic.add_argument(
        "--camera-id",
        choices=("FillCam", "CapSealCam"),
        default="FillCam",
        help="renderer camera identity; defaults to the FillCam ROI assumption",
    )
    synthetic.add_argument("--output-root", default=".")
    synthetic.add_argument("--features-config", default="configs/features.yaml")
    synthetic.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inputs and write a report without extracting or measuring frames",
    )
    synthetic.set_defaults(handler=command_process_synthetic)
    run = subparsers.add_parser("run", help="run the validated Video 5 workflow end to end")
    run.add_argument("--config", default="configs/dataset.yaml")
    run.add_argument("--example-vial", type=int, default=30)
    run.add_argument("--skip-audit", action="store_true")
    run.add_argument("--skip-hashes", action="store_true")
    run.set_defaults(handler=command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
