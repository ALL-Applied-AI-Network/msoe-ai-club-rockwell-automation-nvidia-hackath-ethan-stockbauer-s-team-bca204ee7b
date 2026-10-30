"""Train a dependency-free logistic baseline on a transition manifest.

This is a sanity check before adding an image encoder or TCN. It uses only
metadata and color-error features, and evaluates by the manifest's
transition-level train/validation/test split.

Example:
    python tools/train_transition_baseline.py \
        --manifest dataset/recipe_transitions/manifest.jsonl
"""
import argparse
import json
import math
from pathlib import Path


def load_records(path):
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"{path} is empty")
    required = {
        "split", "transition_id", "vial_index_after_switch", "recipe_before",
        "recipe_after", "expected_color_srgb", "observed_color_srgb",
        "expected_volume_ml", "simulated_volume_ml", "rejected",
    }
    missing = required.difference(records[0])
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return records


def features(record):
    expected = record["expected_color_srgb"]
    observed = record["observed_color_srgb"]
    color_delta = [observed[channel] - expected[channel] for channel in range(3)]
    return [
        1.0,
        float(record["vial_index_after_switch"]),
        float(record["recipe_before"]) / 20.0,
        float(record["recipe_after"]) / 20.0,
        *expected,
        *observed,
        *color_delta,
        float(record["color_error"]),
        float(record["simulated_volume_ml"]) / max(float(record["expected_volume_ml"]), 1e-9) - 1.0,
    ]


def sigmoid(value):
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def fit_logistic(rows, epochs, learning_rate, l2):
    weights = [0.0] * len(rows[0][0])
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for vector, target in rows:
            probability = sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
            error = probability - target
            for index, value in enumerate(vector):
                gradient[index] += error * value
        scale = 1.0 / len(rows)
        for index in range(len(weights)):
            penalty = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= learning_rate * (gradient[index] * scale + penalty)
    return weights


def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = sum(rank + 1 for rank, index in enumerate(order) if labels[index])
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(rows, weights, threshold):
    labels = [target for _, target in rows]
    scores = [
        sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
        for vector, _ in rows
    ]
    predicted = [score >= threshold for score in scores]
    true_positive = sum(label and prediction for label, prediction in zip(labels, predicted))
    false_positive = sum(not label and prediction for label, prediction in zip(labels, predicted))
    false_negative = sum(label and not prediction for label, prediction in zip(labels, predicted))
    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    return {
        "examples": len(rows),
        "positives": sum(labels),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(sum(prediction == bool(label) for label, prediction in zip(labels, predicted)) / len(labels), 4),
        "auc": None if auc(labels, scores) is None else round(auc(labels, scores), 4),
    }


def make_rows(records, target):
    by_transition = {}
    for record in records:
        by_transition.setdefault(record["transition_id"], []).append(record)
    rows = []
    for transition_records in by_transition.values():
        by_vial = {}
        for record in transition_records:
            by_vial.setdefault(record["vial"], record)
        vials = sorted(
            by_vial.values(),
            key=lambda record: record["vial_index_after_switch"],
        )
        for index, record in enumerate(vials):
            if target == "current":
                label = bool(record["rejected"])
            else:
                if index + 1 >= len(vials):
                    continue
                label = bool(vials[index + 1]["rejected"])
            rows.append((features(record), int(label)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", choices=("current", "next"), default="next")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.l2 < 0:
        parser.error("epochs must be positive; learning-rate must be positive; l2 cannot be negative")

    records = load_records(args.manifest.expanduser().resolve())
    splits = {}
    for record in records:
        splits.setdefault(record["split"], []).append(record)
    if not {"train", "validation", "test"}.issubset(splits):
        raise ValueError("manifest must contain train, validation, and test splits")

    train_rows = make_rows(splits["train"], args.target)
    validation_rows = make_rows(splits["validation"], args.target)
    test_rows = make_rows(splits["test"], args.target)
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("each split needs usable rows for the selected target")
    weights = fit_logistic(train_rows, args.epochs, args.learning_rate, args.l2)
    result = {
        "target": args.target,
        "features": len(weights),
        "train": evaluate(train_rows, weights, args.threshold),
        "validation": evaluate(validation_rows, weights, args.threshold),
        "test": evaluate(test_rows, weights, args.threshold),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
