import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from src.perceptron import Perceptron
from src.metrics import evaluate_binary


ROOT = Path(__file__).resolve().parent


def format_metric(value):
    return f"{value:.4f}" if value is not None else "undefined"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "processed_data",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    paths = {
        name: data_dir / f"{name}.npy"
        for name in ["X_train", "X_test", "y_train", "y_test"]
    }
    metadata_file = data_dir / "metadata.json"

    required_files = [*paths.values(), metadata_file]
    missing_files = [path for path in required_files if not path.is_file()]

    if missing_files:
        print("WAITING FOR PROCESSED DATA")
        for path in missing_files:
            print("Missing:", path)
        return

    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_mapping = {"0": "normal", "1": "attack"}

    if metadata.get("label_mapping") != expected_mapping:
        raise ValueError(
            "Confirm the label mapping: 0 = normal, 1 = attack."
        )

    arrays = {
        name: np.load(path, allow_pickle=False)
        for name, path in paths.items()
    }

    X_train = arrays["X_train"]
    X_test = arrays["X_test"]
    y_train = arrays["y_train"]
    y_test = arrays["y_test"]

    for name, X, y in [
        ("Training", X_train, y_train),
        ("Test", X_test, y_test),
    ]:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} features must be a nonempty 2D array.")

        if y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"{name} features and labels do not match.")

        if not np.isfinite(X).all():
            raise ValueError(f"{name} features contain NaN or infinity.")

        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{name} labels must be 0 or 1.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test feature counts must match.")

    print("RAW-FEATURE PERCEPTRON BASELINE")
    print("Random seed:", args.seed)
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Training...", flush=True)

    model = Perceptron(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        random_state=args.seed,
    )

    start = perf_counter()
    model.train(X_train, y_train)
    training_seconds = perf_counter() - start

    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)

    if not np.isfinite(decision_scores).all():
        raise ValueError("Model produced non-finite decision scores.")

    results = evaluate_binary(y_test, predictions)

    results.update({
        "model": "raw_feature_perceptron",
        "evaluation": "standard_test",
        "data_directory": str(data_dir),
        "label_mapping": expected_mapping,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "maximum_epochs": args.epochs,
        "epochs_completed": len(model.mistakes_per_epoch),
        "updates_per_epoch": model.mistakes_per_epoch,
        "training_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "training_seconds": training_seconds,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "package_versions": {
            package: version(package)
            for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
    })

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    run_name = f"raw_baseline_seed{args.seed}"
    metrics_file = output_dir / f"{run_name}.json"
    predictions_file = output_dir / f"{run_name}_predictions.npz"

    np.savez_compressed(
        predictions_file,
        y_true=y_test,
        y_pred=predictions,
        decision_scores=decision_scores,
    )

    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    print("Epochs completed:", results["epochs_completed"])
    print(f"Training seconds: {training_seconds:.3f}")
    print("Accuracy:", format_metric(results["accuracy"]))
    print("Precision:", format_metric(results["precision"]))
    print("Recall:", format_metric(results["recall"]))
    print(
        "False-positive rate:",
        format_metric(results["false_positive_rate"]),
    )
    print("Confusion matrix:")

    for row in results["confusion_matrix"]:
        print(row)

    print("Metrics saved to:", metrics_file)
    print("Predictions saved to:", predictions_file)


if __name__ == "__main__":
    main()