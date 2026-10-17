import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.metrics import evaluate_binary


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
FIGURES = ROOT / "report" / "figures"
METRICS = ["accuracy", "precision", "recall", "false_positive_rate", "roc_auc", "training_seconds"]
COLORS = ["#2563eb", "#059669", "#c026d3"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_runs(result_folder, data_folder, seeds):
    expected_y = np.load(data_folder / "y_test.npy", allow_pickle=False)
    runs = []
    for seed in seeds:
        name = f"raw_baseline_seed{seed}"
        record = json.loads((result_folder / f"{name}.json").read_text(encoding="utf-8"))
        with np.load(result_folder / f"{name}_predictions.npz", allow_pickle=False) as stored:
            true = stored["y_true"].copy()
            predicted = stored["y_pred"].copy()
            scores = stored["decision_scores"].copy()
        assert record["seed"] == seed
        assert np.array_equal(true, expected_y)
        assert true.shape == predicted.shape == scores.shape
        assert np.isfinite(scores).all()
        assert np.array_equal(predicted, (scores >= 0).astype(np.int64))
        checked = evaluate_binary(true, predicted)
        for key, value in checked.items():
            assert record[key] == value, f"Stored metric mismatch: {name}, {key}"
        record["roc_auc"] = float(roc_auc_score(true, scores))
        runs.append({"record": record, "y_true": true, "y_pred": predicted, "scores": scores})
    return runs


def summarize(runs):
    records = [run["record"] for run in runs]
    summary = {
        "seeds": [record["seed"] for record in records],
        "n_runs": len(records),
        "training_shape": records[0]["training_shape"],
        "test_shape": records[0]["test_shape"],
        "metrics": {},
        "confusion_matrix_seed42": next(record["confusion_matrix"] for record in records if record["seed"] == 42),
        "all_reached_epoch_limit": all(record["epochs_completed"] == record["maximum_epochs"] for record in records),
    }
    for metric in METRICS:
        values = np.asarray([record[metric] for record in records if record[metric] is not None], dtype=float)
        summary["metrics"][metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_defined": len(values),
        }
    return summary


def save_figure(figure, name):
    output = PLOTS / name
    figure.savefig(output, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    shutil.copy2(output, FIGURES / name)


def plot_roc(scenarios):
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")
    for axis, (label, runs) in zip(axes, scenarios.items()):
        for color, run in zip(COLORS, runs):
            fpr, tpr, _ = roc_curve(run["y_true"], run["scores"])
            record = run["record"]
            axis.plot(fpr, tpr, color=color, linewidth=1.7,
                      label=f"Seed {record['seed']} (AUC {record['roc_auc']:.3f})")
            axis.scatter(record["false_positive_rate"], record["recall"], color=color,
                         s=25, zorder=3)
        axis.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1, label="Chance")
        axis.set(title=label, xlabel="False-positive rate", ylabel="Attack recall",
                 xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.18)
        axis.legend(loc="lower right", fontsize=8)
    save_figure(figure, "baseline_roc.png")


def plot_confusion(scenarios, attack):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for axis, (label, runs) in zip(axes, scenarios.items()):
        run = next(run for run in runs if run["record"]["seed"] == 42)
        matrix = np.asarray(run["record"]["confusion_matrix"], dtype=int)
        row_fractions = matrix / matrix.sum(axis=1, keepdims=True)
        artist = axis.imshow(row_fractions, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}\n{row_fractions[row, column]:.1%}",
                          ha="center", va="center", fontsize=11,
                          color="white" if row_fractions[row, column] > 0.55 else "#0f172a")
        actual_labels = ["Normal", "Attack"] if label.startswith("Standard") else ["Normal", attack]
        axis.set(title=f"{label}\nSeed 42", xlabel="Predicted class", ylabel="Actual class",
                 xticks=[0, 1], yticks=[0, 1])
        axis.set_xticklabels(["Normal", "Attack"])
        axis.set_yticklabels(actual_labels)
    colorbar = figure.colorbar(artist, ax=list(axes), fraction=0.035, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("Share of actual class")
    save_figure(figure, "baseline_confusion_matrices.png")


def plot_updates(scenarios):
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
    for axis, (label, runs) in zip(axes, scenarios.items()):
        for color, run in zip(COLORS, runs):
            updates = run["record"]["updates_per_epoch"]
            axis.plot(np.arange(1, len(updates) + 1), updates, color=color, linewidth=1.2,
                      label=f"Seed {run['record']['seed']}")
        axis.set(title=label, xlabel="Epoch", ylabel="Online weight updates")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    save_figure(figure, "baseline_training_updates.png")


def verify_heldout(data, attack):
    from preprocess_data import CATEGORICAL, NUMERIC, load_kdd

    raw_train = load_kdd(ROOT / "data" / "KDDTrain+.txt")
    raw_test = load_kdd(ROOT / "data" / "KDDTest+.txt")
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    train_rows = np.load(data / "train_source_rows.npy", allow_pickle=False)
    test_rows = np.load(data / "test_source_rows.npy", allow_pickle=False)
    expected_train = raw_train.index[raw_train["label"] != attack].to_numpy()
    expected_test = raw_test.index[raw_test["label"].isin(["normal", attack])].to_numpy()
    assert np.array_equal(train_rows, expected_train)
    assert np.array_equal(test_rows, expected_test)
    assert np.array_equal(np.load(data / "train_attack_labels.npy", allow_pickle=False),
                          raw_train.loc[train_rows, "label"].to_numpy(dtype=str))
    assert np.array_equal(np.load(data / "test_attack_labels.npy", allow_pickle=False),
                          raw_test.loc[test_rows, "label"].to_numpy(dtype=str))
    X = np.load(data / "X_train.npy", mmap_mode="r", allow_pickle=False)
    means = X[:, :len(NUMERIC)].mean(axis=0, dtype=np.float64)
    variances = X[:, :len(NUMERIC)].var(axis=0, dtype=np.float64)
    assert np.max(np.abs(means)) < 1e-5
    assert np.max(np.minimum(np.abs(variances), np.abs(variances - 1))) < 1e-5
    for name in CATEGORICAL:
        indices = [i for i, feature in enumerate(metadata["feature_names"]) if feature.startswith(f"categorical__{name}_")]
        assert (X[:, indices].sum(axis=1) == 1).all()
    return {
        "status": "passed",
        "held_out_attack": attack,
        "source_row_membership_checked": True,
        "heldout_attack_absent_from_training": True,
        "evaluation_uses_original_test_file": True,
        "reduced_training_numeric_scaling_checked": True,
        "maximum_absolute_numeric_training_mean": float(np.max(np.abs(means))),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
    }


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    seeds = config["seeds"]
    attack = config["held_out"]["attack_type"]
    assert len(seeds) >= 3 and len(seeds) == len(set(seeds))
    PLOTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11})

    standard = read_runs(RESULTS, ROOT / config["data_directory"], seeds)
    heldout_data = ROOT / "processed_data" / "heldout" / attack
    heldout = read_runs(RESULTS / "heldout" / attack, heldout_data, seeds)
    verification = verify_heldout(heldout_data, attack)
    preprocessing_checks = json.loads((RESULTS / "preprocessing_validation.json").read_text(encoding="utf-8"))
    for filename, expected in preprocessing_checks["sha256"].items():
        assert sha256(ROOT / "processed_data" / filename) == expected

    scenarios = {"Standard NSL-KDD test": standard, f"Held-out {attack}": heldout}
    summary = {
        "seeds": seeds,
        "aggregation": "Arithmetic mean and sample standard deviation (ddof=1); variability across seeds on a fixed test split, not a confidence interval",
        "scenarios": {label: summarize(runs) for label, runs in scenarios.items()},
        "heldout_verification": verification,
        "input_and_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [ROOT / "experiment_config.json", ROOT / "src" / "perceptron.py",
                         ROOT / "src" / "metrics.py", ROOT / "preprocess_data.py",
                         ROOT / "train_baseline.py", ROOT / "run_heldout_baselines.py",
                         ROOT / "processed_data" / "metadata.json", heldout_data / "metadata.json"]
        },
    }
    (RESULTS / "baseline_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    all_rows = []
    summary_rows = []
    for label, runs in scenarios.items():
        for run in runs:
            record = run["record"]
            all_rows.append({"evaluation": label, "seed": record["seed"], "n_samples": record["n_samples"],
                             **{metric: record[metric] for metric in METRICS}})
        for metric, values in summary["scenarios"][label]["metrics"].items():
            summary_rows.append({"evaluation": label, "metric": metric, **values})
    for filename, rows in [("baseline_metrics_all_runs.csv", all_rows), ("baseline_metric_summary.csv", summary_rows)]:
        with (RESULTS / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plot_roc(scenarios)
    plot_confusion(scenarios, attack)
    plot_updates(scenarios)
    print("BASELINE ANALYSIS AND OUTPUT VERIFICATION COMPLETED")
    for label, entry in summary["scenarios"].items():
        print(label)
        for metric in METRICS:
            stats = entry["metrics"][metric]
            print(f"  {metric}: mean={stats['mean']:.6f}, sample SD={stats['sample_sd']:.6f}, n={stats['n_defined']}")
    print("Saved summary, CSV tables and three plots to:", RESULTS)


if __name__ == "__main__":
    main()
