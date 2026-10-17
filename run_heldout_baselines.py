import argparse
import csv
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from src.metrics import evaluate_binary
from src.perceptron import Perceptron


ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default=config["held_out"]["attack_type"])
    args = parser.parse_args()
    if not args.attack or not all(c.isalnum() or c == "_" for c in args.attack):
        raise ValueError("Invalid attack label.")

    data = ROOT / "processed_data" / "heldout" / args.attack
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["held_out_attack"] == args.attack
    assert metadata["preprocessing_fit"] == "reduced_training_only"
    assert metadata["label_mapping"] == {"0": "normal", "1": "attack"}
    for filename, expected in metadata["processed_sha256"].items():
        with (data / filename).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected

    X_train = np.load(data / "X_train.npy", allow_pickle=False)
    X_test = np.load(data / "X_test.npy", allow_pickle=False)
    y_train = np.load(data / "y_train.npy", allow_pickle=False)
    y_test = np.load(data / "y_test.npy", allow_pickle=False)
    mask = np.load(data / "heldout_attack_mask.npy", allow_pickle=False)
    train_names = np.load(data / "train_attack_labels.npy", allow_pickle=False)
    test_names = np.load(data / "test_attack_labels.npy", allow_pickle=False)
    test_rows = np.load(data / "test_source_rows.npy", allow_pickle=False)
    assert not (train_names == args.attack).any()
    assert np.array_equal(y_train, (train_names != "normal").astype(np.int64))
    assert np.array_equal(mask, test_names == args.attack)
    assert np.array_equal(y_test, mask.astype(np.int64))
    assert mask.any() and (~mask).any()
    assert X_train.shape[0] == len(y_train) and X_test.shape[0] == len(y_test)
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()

    output = ROOT / "results" / "heldout" / args.attack
    output.mkdir(parents=True, exist_ok=True)
    settings = config["perceptron"]
    rows = []
    for seed in config["seeds"]:
        print(f"\nHELD-OUT {args.attack}: BASELINE SEED {seed}", flush=True)
        model = Perceptron(
            learning_rate=settings["learning_rate"],
            epochs=settings["epochs"],
            random_state=seed,
        )
        start = perf_counter()
        model.train(X_train, y_train)
        seconds = perf_counter() - start
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)
        assert np.isfinite(scores).all()
        result = evaluate_binary(y_test, predictions)
        result.update({
            "model": "raw_feature_perceptron",
            "evaluation": "heldout_attack",
            "held_out_attack": args.attack,
            "seed": seed,
            "learning_rate": settings["learning_rate"],
            "maximum_epochs": settings["epochs"],
            "epochs_completed": len(model.mistakes_per_epoch),
            "updates_per_epoch": model.mistakes_per_epoch,
            "training_seconds": seconds,
            "training_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "heldout_attack_samples": int(mask.sum()),
            "normal_test_samples": int((~mask).sum()),
            "heldout_attack_recall": float((predictions[mask] == 1).mean()),
            "normal_false_positive_rate": float((predictions[~mask] == 1).mean()),
            "preprocessing_fit": "reduced_training_only",
            "hyperparameter_selection": "Fixed configuration; held-out attack not used for selection",
            "data_directory": str(data),
            "data_sha256": metadata["processed_sha256"],
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "package_versions": {name: version(name) for name in ["numpy", "scikit-learn"]},
        })
        assert result["recall"] == result["heldout_attack_recall"]
        assert result["false_positive_rate"] == result["normal_false_positive_rate"]
        name = f"raw_baseline_seed{seed}"
        np.savez_compressed(
            output / f"{name}_predictions.npz",
            y_true=y_test, y_pred=predictions, decision_scores=scores,
            heldout_attack_mask=mask, source_test_rows=test_rows,
        )
        (output / f"{name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        fields = ["seed", "n_samples", "accuracy", "precision", "recall", "false_positive_rate", "training_seconds", "epochs_completed"]
        rows.append({field: result[field] for field in fields})
        print("Accuracy:", f"{result['accuracy']:.4f}")
        print("Held-out attack recall:", f"{result['heldout_attack_recall']:.4f}")
        print("Normal false-positive rate:", f"{result['normal_false_positive_rate']:.4f}")
        print("Confusion matrix:", result["confusion_matrix"])
        print("Training seconds:", f"{seconds:.3f}", flush=True)

    with (output / "raw_baseline_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\nHELD-OUT BASELINE RUNS COMPLETED:", output, flush=True)


if __name__ == "__main__":
    main()
