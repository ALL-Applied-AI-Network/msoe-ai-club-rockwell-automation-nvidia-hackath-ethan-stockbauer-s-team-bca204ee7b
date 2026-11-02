"""Train a causal TCN on the frame manifest.

The model consumes compact per-frame vial appearance features rather than
future-aware image embeddings. This keeps the first training run lightweight
and makes the same feature extractor usable for unlabeled Video 5 inference.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_FEATURE_SIZE = 19
RECIPE_FEATURE_SIZE = 20
APPROVED_COLOR_MODEL = "beer_lambert_rgb_v6_video5_r05_r06_opacity_075"
APPROVED_LIQUID_OPACITY = 0.75
APPROVED_RECIPES = {5, 6}


def frame_features_from_image(image, source="<image>"):
    if image is None:
        raise ValueError(f"could not read frame: {source}")
    height, width = image.shape[:2]
    crop = image[
        round(height * 366 / 480):round(height * 407 / 480),
        round(width * 294 / 640):round(width * 355 / 640),
    ]
    if crop.size == 0:
        raise ValueError(f"frame is too small for the calibrated liquid ROI: {source}")
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_normalized = hsv / np.array([180.0, 255.0, 255.0], dtype=np.float32)
    colored = hsv[:, :, 1] > 35
    colored_rgb = rgb[colored]
    colored_hsv = hsv_normalized[colored]
    if not len(colored_rgb):
        colored_rgb = np.zeros((1, 3), dtype=np.float32)
        colored_hsv = np.zeros((1, 3), dtype=np.float32)
    return np.array([
        *rgb.mean(axis=(0, 1)),
        *rgb.std(axis=(0, 1)),
        *hsv_normalized.mean(axis=(0, 1)),
        *hsv_normalized.std(axis=(0, 1)),
        float(colored.mean()),
        *colored_rgb.mean(axis=0),
        *colored_hsv.mean(axis=0),
    ], dtype=np.float32)


def frame_features(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read frame: {path}")
    return frame_features_from_image(image, path)


def recipe_features(record):
    values = []
    for recipe in (record["recipe_before"], record["recipe_after"]):
        values.extend(float(recipe == candidate) for candidate in (5, 6))
    values.extend(float(volume) / 100.0 for volume in record["tank_volumes_ml"])
    values.append(sum(record["tank_volumes_ml"]) / 100.0)
    order = {tank: index for index, tank in enumerate(record["dispense_order"])}
    for tank in ("T100", "T200", "T400"):
        values.extend(float(order.get(tank) == position) for position in range(3))
    values.extend(float(record["carryover_tank"] == tank) for tank in ("T100", "T200", "T400"))
    if len(values) != RECIPE_FEATURE_SIZE:
        raise ValueError(f"recipe feature vector has {len(values)} values")
    return np.asarray(values, dtype=np.float32)


class ManifestDataset(Dataset):
    def __init__(self, records, root, target, feature_mean, feature_std, recipe_aware):
        self.records = records
        self.root = root
        self.target = target
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.recipe_aware = recipe_aware
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        features = []
        for relative in record["input_frames"]:
            path = self.root / relative
            key = str(path)
            if key not in self.cache:
                self.cache[key] = frame_features(path)
            features.append(self.cache[key])
        target = record[self.target]
        sequence = (np.stack(features, axis=1) - self.feature_mean[:, None]) / self.feature_std[:, None]
        if self.recipe_aware:
            context = np.repeat(recipe_features(record)[:, None], sequence.shape[1], axis=1)
            sequence = np.concatenate((sequence, context), axis=0)
        return torch.from_numpy(sequence.astype(np.float32)), torch.tensor(float(target))


def fit_feature_normalization(records, root):
    paths = sorted({root / record["input_frames"][-1] for record in records})
    values = np.stack([frame_features(path) for path in paths])
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


class CausalTCN(nn.Module):
    def __init__(self, input_size=IMAGE_FEATURE_SIZE, channels=32):
        super().__init__()
        self.input = nn.Conv1d(input_size, channels, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, dilation=dilation)
            for dilation in (1, 2, 4, 8)
        ])
        self.output = nn.Linear(channels, 1)

    def forward(self, inputs):
        x = self.input(inputs)
        for block in self.blocks:
            padding = 2 * block.dilation[0]
            x = nn.functional.pad(x, (padding, 0))
            x = torch.relu(block(x))
        return self.output(x[:, :, -1]).squeeze(1)


def load_records(path):
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"{path} is empty")
    unverified = [
        record for record in records
        if record.get("color_model") != APPROVED_COLOR_MODEL
        or record.get("liquid_opacity") != APPROVED_LIQUID_OPACITY
        or record.get("recipe_before") not in APPROVED_RECIPES
        or record.get("recipe_after") not in APPROVED_RECIPES
    ]
    if unverified:
        examples = sorted({record.get("clip_id", "<unknown>") for record in unverified})[:5]
        raise ValueError(
            f"manifest contains {len(unverified)} records without the approved "
            f"recipe 5/6 color calibration; example clips: {', '.join(examples)}"
        )
    return records


def binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    positives = int(labels.sum())
    negatives = len(labels) - positives
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    true_negative = negatives - false_positive
    false_negative = positives - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, positives)
    specificity = true_negative / max(1, negatives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    average_precision = None
    roc_auc = None
    if positives and negatives:
        order = np.argsort(-scores, kind="stable")
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        cumulative_positive = np.cumsum(sorted_labels)
        cumulative_total = np.arange(1, len(labels) + 1)
        average_precision = float(
            np.sum((cumulative_positive / cumulative_total) * sorted_labels) / positives
        )

        ascending = np.argsort(scores, kind="stable")
        sorted_ascending = scores[ascending]
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_ascending[end] == sorted_ascending[start]:
                end += 1
            ranks[ascending[start:end]] = (start + 1 + end) / 2
            start = end
        positive_rank_sum = ranks[labels == 1].sum()
        roc_auc = float(
            (positive_rank_sum - positives * (positives + 1) / 2)
            / (positives * negatives)
        )

    return {
        "accuracy": round(float(np.mean(predictions == labels)), 4),
        "majority_baseline_accuracy": round(max(positives, negatives) / max(1, len(labels)), 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_precision": round(average_precision, 4) if average_precision is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "brier_score": round(float(np.mean((scores - labels) ** 2)), 6),
        "positive_examples": positives,
        "negative_examples": negatives,
        "examples": len(labels),
    }


def collect_predictions(model, loader, device, positive_weight):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses, labels, scores = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            losses.append(loss_fn(logits, targets.to(device)).item())
            labels.extend(targets.numpy().tolist())
            calibrated_logits = logits - math.log(positive_weight)
            scores.extend(torch.sigmoid(calibrated_logits).cpu().numpy().tolist())
    return labels, scores, round(float(np.mean(losses)), 6)


def select_f1_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_true_positive = np.cumsum(sorted_labels)
    cumulative_false_positive = np.cumsum(1 - sorted_labels)
    group_ends = np.flatnonzero(
        np.r_[np.diff(sorted_scores) != 0, True]
    )
    best_f1 = -1.0
    best_threshold = 0.5
    for end in group_ends:
        true_positive = cumulative_true_positive[end]
        false_positive = cumulative_false_positive[end]
        false_negative = int(labels.sum()) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        threshold = (
            float((sorted_scores[end] + sorted_scores[end + 1]) / 2)
            if end + 1 < len(sorted_scores)
            else float(sorted_scores[end])
        )
        if f1 > best_f1 or (f1 == best_f1 and threshold < best_threshold):
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def aggregate_clip_predictions(records, scores, decision_window_seconds=2.0):
    clips = {}
    for record, score in zip(records, scores):
        clips.setdefault(record["clip_id"], []).append((record, score))
    labels, clip_scores = [], []
    for items in clips.values():
        items.sort(key=lambda item: item[0]["frame_index"])
        first = items[0][0]
        onset = first["contamination_onset_frame"]
        onset_record = next(
            (record for record, _ in items if record["frame_index"] == onset),
            None,
        )
        if onset_record is None:
            if "target_prefill_reject" not in first:
                continue
            labels.append(int(first["target_prefill_reject"]))
            clip_scores.append(max(score for _, score in items))
            continue
        end_time = onset_record["timestamp_s"] + decision_window_seconds
        window_scores = [
            score for record, score in items
            if onset <= record["frame_index"]
            and record["timestamp_s"] <= end_time
        ]
        if not window_scores:
            continue
        labels.append(int(first["contamination_ml"] > 0))
        clip_scores.append(float(np.mean(window_scores)))
    return labels, clip_scores


def evaluate(model, loader, device, positive_weight, threshold=0.5):
    labels, scores, loss = collect_predictions(model, loader, device, positive_weight)
    return {
        "loss": loss,
        **binary_metrics(labels, scores, threshold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("target_current_reject", "target_future_reject", "target_prefill_reject"), default="target_future_reject")
    parser.add_argument("--recipe-aware", action="store_true",
                        help="include recipe, dispense-order, and carryover context features")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-records", type=int, default=0,
                        help="cap training records; 0 uses the full train split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_train_records < 0:
        parser.error("epochs and batch-size must be positive; max-train-records cannot be negative")
    torch.manual_seed(7)
    manifest = args.manifest.expanduser().resolve()
    records = load_records(manifest)
    splits = {name: [r for r in records if r["split"] == name] for name in ("train", "validation", "test")}
    if any(not rows for rows in splits.values()):
        raise ValueError("manifest must contain records in train, validation, and test splits")
    train_records = (
        splits["train"][:args.max_train_records]
        if args.max_train_records
        else splits["train"]
    )
    train_positive = sum(record[args.target] for record in train_records)
    train_negative = len(train_records) - train_positive
    if not train_positive or not train_negative:
        raise ValueError(
            f"training split must contain both classes; got "
            f"{train_positive} positive and {train_negative} negative records"
        )
    root = manifest.parent
    feature_mean, feature_std = fit_feature_normalization(train_records, root)
    datasets = {
        name: ManifestDataset(rows, root, args.target, feature_mean, feature_std, args.recipe_aware)
        for name, rows in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    loaders["train"] = DataLoader(
        ManifestDataset(train_records, root, args.target, feature_mean, feature_std, args.recipe_aware),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch environment")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model = CausalTCN(
        input_size=IMAGE_FEATURE_SIZE + (RECIPE_FEATURE_SIZE if args.recipe_aware else 0)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    positive_weight = train_negative / train_positive
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    best_average_precision = -1.0
    best_epoch = 0
    best_state = None
    best_validation = None
    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        validation = evaluate(
            model, loaders["validation"], device, positive_weight
        )
        print(json.dumps({"epoch": epoch + 1, "validation": validation}), flush=True)
        average_precision = validation["average_precision"]
        if average_precision is not None and average_precision > best_average_precision:
            best_average_precision = average_precision
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_validation = validation
    if best_state is None:
        raise RuntimeError("validation could not select a checkpoint because it has no positive/negative pair")
    model.load_state_dict(best_state)
    validation_labels, validation_scores, _ = collect_predictions(
        model, loaders["validation"], device, positive_weight
    )
    decision_threshold = select_f1_threshold(
        validation_labels, validation_scores
    )
    test_labels, test_scores, _ = collect_predictions(
        model, loaders["test"], device, positive_weight
    )
    validation_clip_labels, validation_clip_scores = aggregate_clip_predictions(
        splits["validation"], validation_scores
    )
    test_clip_labels, test_clip_scores = aggregate_clip_predictions(
        splits["test"], test_scores
    )
    clip_decision_threshold = select_f1_threshold(
        validation_clip_labels, validation_clip_scores
    )
    result = {
        "target": args.target,
        "device": str(device),
        "data_quality_gate": "video5_recipe_5_6_v6_opacity_075",
        "color_models": sorted({record["color_model"] for record in records}),
        "training_batches": sorted({record["clip_id"].split("/")[0] for record in train_records}),
        "training_recipes": sorted({
            recipe
            for record in train_records
            for recipe in (record["recipe_before"], record["recipe_after"])
        }),
        "train_records": len(train_records),
        "train_positive_records": train_positive,
        "train_negative_records": train_negative,
        "positive_class_weight": round(positive_weight, 6),
        "best_epoch": best_epoch,
        "validation_at_0_5": best_validation,
        "decision_threshold": round(decision_threshold, 6),
        "validation": evaluate(
            model, loaders["validation"], device, positive_weight, decision_threshold
        ),
        "test": evaluate(
            model, loaders["test"], device, positive_weight, decision_threshold
        ),
        "clip_level_first_2s_after_visible_onset": {
            "decision_threshold": round(clip_decision_threshold, 6),
            "validation": binary_metrics(
                validation_clip_labels, validation_clip_scores, clip_decision_threshold
            ),
            "test": binary_metrics(
                test_clip_labels, test_clip_scores, clip_decision_threshold
            ),
        },
    }
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": result,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
