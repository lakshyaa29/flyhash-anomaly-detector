"""Check published metrics and summary tables without downloading the dataset."""

import csv
import json
from pathlib import Path

import numpy as np

from src.metrics import evaluate_binary


ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    root = Path(root)
    summary = json.loads((root / "results/baseline_summary.json").read_text(encoding="utf-8"))
    config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
    attack = config["held_out"]["attack_type"]
    with (root / "results/baseline_metrics_all_runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 * len(summary["seeds"]):
        raise ValueError("Published table does not contain all expected runs.")
    with (root / "results/baseline_metric_summary.csv").open(encoding="utf-8", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    expected_pairs = {(label, metric) for label, scenario in summary["scenarios"].items()
                      for metric in scenario["metrics"]}
    if len(summary_rows) != len(expected_pairs) or {(row["evaluation"], row["metric"]) for row in summary_rows} != expected_pairs:
        raise ValueError("Metric summary CSV does not cover all scenarios and metrics.")
    for label, scenario in summary["scenarios"].items():
        folder = root / "results" if label.startswith("Standard") else root / "results/heldout" / attack
        selected = [row for row in rows if row["evaluation"] == label]
        if {int(row["seed"]) for row in selected} != set(summary["seeds"]):
            raise ValueError(f"Seed mismatch in {label}.")
        for row in selected:
            record = json.loads((folder / f"raw_baseline_seed{row['seed']}.json").read_text(encoding="utf-8"))
            if record["seed"] != int(row["seed"]) or record["learning_rate"] != config["perceptron"]["learning_rate"] or record["maximum_epochs"] != config["perceptron"]["epochs"]:
                raise ValueError("Recorded run settings disagree with the experiment configuration.")
            matrix = np.asarray(record["confusion_matrix"])
            if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all():
                raise ValueError("Confusion counts must be nonnegative integers.")
            tn, fp, fn, tp = map(int, matrix.ravel())
            true = [0] * (tn + fp) + [1] * (fn + tp)
            pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
            checked = evaluate_binary(true, pred)
            for key, expected in checked.items():
                if record[key] != expected:
                    raise ValueError(f"Recorded metric mismatch: {label}, seed {row['seed']}, {key}.")
            for metric in ["accuracy", "precision", "recall", "false_positive_rate", "training_seconds"]:
                value = None if row[metric] == "" else float(row[metric])
                if value != record[metric]:
                    raise ValueError(f"CSV and JSON disagree: {metric}.")
            if not 0 <= float(row["roc_auc"]) <= 1:
                raise ValueError("Invalid ROC AUC.")
        for metric, stats in scenario["metrics"].items():
            csv_stats = next(row for row in summary_rows if row["evaluation"] == label and row["metric"] == metric)
            if int(csv_stats["n_defined"]) != stats["n_defined"]:
                raise ValueError("Metric CSV defined-value count mismatch.")
            for key in ["mean", "sample_sd"]:
                csv_value = None if csv_stats[key] == "" else float(csv_stats[key])
                if csv_value != stats[key]:
                    raise ValueError(f"Metric CSV and summary JSON disagree: {label}, {metric}, {key}.")
            values = np.asarray([float(row[metric]) for row in selected if row[metric] != ""], dtype=float)
            if len(values) != stats["n_defined"]:
                raise ValueError("Defined-value count mismatch.")
            mean = float(values.mean()) if len(values) else None
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            for observed, expected in [(mean, stats["mean"]), (sd, stats["sample_sd"])]:
                if observed is None or expected is None:
                    if observed != expected:
                        raise ValueError("Undefined summary value mismatch.")
                elif not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Summary arithmetic mismatch for {metric}.")
    print(f"RECORDED RESULT CHECKS PASSED: {len(rows)} runs and both summary tables.")
    return len(rows)


if __name__ == "__main__":
    verify()
