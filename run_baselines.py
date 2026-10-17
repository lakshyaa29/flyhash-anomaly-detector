import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    config_file = ROOT / "experiment_config.json"

    with config_file.open("r", encoding="utf-8") as file:
        config = json.load(file)

    data_dir = ROOT / config["data_directory"]

    required_names = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "metadata.json",
    ]

    missing = [
        name
        for name in required_names
        if not (data_dir / name).is_file()
    ]

    if missing:
        print("WAITING FOR PROCESSED DATA")
        for name in missing:
            print("Missing:", data_dir / name)
        return

    settings = config["perceptron"]
    results_dir = ROOT / "results"

    fields = [
        "seed",
        "n_samples",
        "accuracy",
        "precision",
        "recall",
        "false_positive_rate",
        "training_seconds",
        "epochs_completed",
    ]

    rows = []

    for seed in config["seeds"]:
        print(f"\nRUNNING BASELINE SEED {seed}", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "train_baseline.py"),
                "--data-dir",
                str(data_dir),
                "--seed",
                str(seed),
                "--learning-rate",
                str(settings["learning_rate"]),
                "--epochs",
                str(settings["epochs"]),
            ],
            cwd=ROOT,
            check=True,
        )

        metrics_file = results_dir / f"raw_baseline_seed{seed}.json"

        with metrics_file.open("r", encoding="utf-8") as file:
            result = json.load(file)

        rows.append({
            field: result[field]
            for field in fields
        })

    output_file = results_dir / "raw_baseline_runs.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nCompleted baseline runs:", len(rows))
    print("Comparison table saved to:", output_file)


if __name__ == "__main__":
    main()