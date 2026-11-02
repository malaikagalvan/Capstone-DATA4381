"""Compare standard vs class-weighted loss on ORIGINAL three-class FER2013.

Run from the cloned repository root:
    python -m capstone_nn.franklin_class_weighting.experiment --data-dir data/final

This experiment imports the team's model classes and image-loader utilities,
but DOES NOT modify the original losses.py, training.py, or dataset files.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from capstone_data.image_loader import (
    CLASS_NAMES, CLASS_TO_INDEX, find_class_folder, find_images,
    load_arrays, stratified_split_indices,
)
from capstone_nn.model import Sequential
from capstone_nn.layers import Flatten, Dense
from capstone_nn.activations import ReLU
from .losses import calculate_class_weights
from .training import train_epoch, evaluate


EXPECTED_CLASSES = ("angry", "happy", "sad")


def original_dataset_records(data_dir):
    """Index unaugmented images in angry/happy/sad folders.

    Works with `data/final/angry` or cleaned folders such as
    `cleaned_data/angry_cleaned`. Augmented sets must NOT be used here.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Original data directory does not exist: {data_dir}")
    if tuple(CLASS_NAMES) != EXPECTED_CLASSES:
        raise ValueError(f"Expected the shared image loader's class order to be {EXPECTED_CLASSES}; got {CLASS_NAMES}")
    records = []
    for name in EXPECTED_CLASSES:
        folder = find_class_folder(name, data_dir, required=True)
        for path in find_images(folder):
            records.append({
                "path": path, "label": CLASS_TO_INDEX[name],
                "class_name": name, "source": "original_only",
            })
    return records


def predict_labels(model, X, batch_size=256):
    outputs = []
    for start in range(0, len(X), batch_size):
        outputs.append(np.argmax(model.forward(X[start:start+batch_size]), axis=1))
    return np.concatenate(outputs)


def model_metrics(y_true, y_pred):
    """Confusion matrix, accuracy, balanced accuracy, per-class and macro F1.

    Rows in confusion matrix are ACTUAL labels; columns are PREDICTED labels.
    Happy-false-positive rate is the share of Angry/Sad images predicted Happy.
    """
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm)
    actual = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(3, dtype=float), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(3, dtype=float), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3, dtype=float), where=(precision + recall) > 0)
    non_happy = y_true != 1
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "happy_false_positive_rate": float(np.mean(y_pred[non_happy] == 1)) if np.any(non_happy) else None,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(actual[i])}
            for i, name in enumerate(EXPECTED_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


def build_model(seed):
    # Both runs start with exactly the same random initialization.
    np.random.seed(seed)
    return Sequential(Flatten(), Dense(2304, 128), ReLU(), Dense(128, 3))


def run_model(name, weights, X_train, y_train, X_val, y_val,
              epochs, batch_size, learning_rate, seed):
    model = build_model(seed)
    rng = np.random.default_rng(seed)
    history = []
    print(f"\n=== {name} ===")
    for epoch in range(1, epochs + 1):
        training = train_epoch(
            model, X_train, y_train, batch_size,
            learning_rate, rng, class_weights=weights,
        )
        validation = evaluate(model, X_val, y_val)
        history.append({"epoch": epoch, "train_loss": training["loss"],
                        "train_accuracy": training["accuracy"],
                        "val_loss": validation["loss"],
                        "val_accuracy": validation["accuracy"]})
        print(f"Epoch {epoch:2d} | train acc {training['accuracy']:.3f} | "
              f"val acc {validation['accuracy']:.3f} | val loss {validation['loss']:.3f}")
    metrics = model_metrics(y_val, predict_labels(model, X_val))
    metrics["validation_unweighted_loss"] = evaluate(model, X_val, y_val)["loss"]
    return model, history, metrics


def save_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/final"),
                        help="Original cleaned dataset root; must contain angry/happy/sad folders")
    parser.add_argument("--output-dir", type=Path, default=Path("class_weighting_results"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-test", action="store_true",
                        help="Evaluate held-out test split ONLY when finalizing experiment")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")

    records = original_dataset_records(args.data_dir)
    X, y = load_arrays(records)
    print("\nOriginal image counts:", dict(zip(EXPECTED_CLASSES, np.bincount(y, minlength=3).tolist())))

    # Split BEFORE any augmentation. This is an independent 70/15/15 split
    # using the shared NumPy helper, NOT guaranteed to match Melissa's saved
    # sklearn split. Align partitions before a formal augmentation comparison.
    train_idx, val_idx, test_idx = stratified_split_indices(
        y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    class_counts = np.bincount(y_train, minlength=3)
    weights = calculate_class_weights(y_train, num_classes=3)
    print("Training class counts:", class_counts.tolist())
    print("Class weights:", dict(zip(EXPECTED_CLASSES, np.round(weights, 4).tolist())))
    print(f"Splits: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)} (test held out)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"data_dir": str(Path(args.data_dir).resolve()),
               "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx),
                         "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15, "seed": args.seed},
               "classes": list(EXPECTED_CLASSES),
               "train_class_counts": class_counts.tolist(),
               "class_weights": weights.tolist(),
               "training_config": {"epochs": args.epochs, "batch_size": args.batch_size,
                                   "learning_rate": args.learning_rate},
               "runs": {}}

    for name, class_weights in (("normal", None), ("weighted", weights)):
        model, history, val_metrics = run_model(
            name, class_weights, X_train, y_train, X_val, y_val,
            args.epochs, args.batch_size, args.learning_rate, args.seed,
        )
        save_history(args.output_dir / f"{name}_training_history.csv", history)
        np.savetxt(args.output_dir / f"{name}_val_confusion_matrix.csv",
                   np.array(val_metrics["confusion_matrix"], dtype=int), fmt="%d", delimiter=",")
        summary["runs"][name] = {"validation": val_metrics}
        print(f"{name}: val accuracy={val_metrics['accuracy']:.3f}, "
              f"macro F1={val_metrics['macro_f1']:.3f}, "
              f"happy false-positive rate={val_metrics['happy_false_positive_rate']:.3f}")
        for label in EXPECTED_CLASSES:
            print(f"  {label}: recall={val_metrics['per_class'][label]['recall']:.3f}, "
                  f"F1={val_metrics['per_class'][label]['f1']:.3f}")
        if args.final_test:
            summary["runs"][name]["test"] = model_metrics(y_test, predict_labels(model, X_test))

    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved experiment results to {args.output_dir.resolve()}")
    if not args.final_test:
        print("Test results were intentionally NOT calculated. Use --final-test only for final reporting.")
    print("NOTE: This script creates its own NumPy 70/15/15 split. For a direct comparison to")
    print("Melissa's augmented models, first ensure the exact original-image partitions match.")


if __name__ == "__main__":
    main()
