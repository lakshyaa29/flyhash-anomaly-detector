"""Run after Siddhesh supplies the documented encoder. No fallback implementation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from preprocess_data import load_kdd, make_preprocessor
from src.encoder_contract import MissingEncoderError, encode_pair, load_encoder_class
from src.metrics import evaluate_binary
from src.perceptron import Perceptron


ROOT = Path(__file__).resolve().parent
VALIDATION_SEED = 2026


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fit_trial(factory, train, evaluation, y_train, y_evaluation, ratio, sparsity, seed, settings):
    encoder = factory(expansion_ratio=ratio, sparsity=sparsity, random_state=seed)
    start = perf_counter()
    encoded_train, encoded_test = encode_pair(encoder, train, evaluation, ratio, sparsity)
    encoding_seconds = perf_counter() - start
    model = Perceptron(learning_rate=settings["learning_rate"], epochs=settings["epochs"], random_state=seed)
    start = perf_counter()
    model.train(encoded_train, y_train)
    training_seconds = perf_counter() - start
    predictions = model.predict(encoded_test)
    scores = model.decision_function(encoded_test)
    result = evaluate_binary(y_evaluation, predictions)
    result.update({
        "seed": seed, "expansion_ratio": ratio, "sparsity": sparsity,
        "active_bits": int(encoded_train[0].sum()), "encoded_features": int(encoded_train.shape[1]),
        "roc_auc": float(roc_auc_score(y_evaluation, scores)),
        "balanced_accuracy": (result["recall"] + 1 - result["false_positive_rate"]) / 2,
        "encoding_seconds": encoding_seconds, "training_seconds": training_seconds,
        "maximum_epochs": settings["epochs"], "learning_rate": settings["learning_rate"],
        "epochs_completed": len(model.mistakes_per_epoch),
    })
    return result, predictions, scores


def scenario_data(scenario, attack):
    raw = load_kdd(ROOT / "data/KDDTrain+.txt")
    if scenario == "heldout":
        raw = raw.loc[raw["label"] != attack].copy()
    y = raw["label"].ne("normal").to_numpy(dtype=np.int64)
    train_rows, valid_rows = train_test_split(np.arange(len(raw)), test_size=0.2,
                                            random_state=VALIDATION_SEED, stratify=y)
    transformer = make_preprocessor()
    features = raw.drop(columns=["label", "difficulty"])
    train = np.asarray(transformer.fit_transform(features.iloc[train_rows]), dtype=np.float32)
    valid = np.asarray(transformer.transform(features.iloc[valid_rows]), dtype=np.float32)
    full_folder = ROOT / "processed_data"
    result_folder = ROOT / "results"
    if scenario == "heldout":
        full_folder = full_folder / "heldout" / attack
        result_folder = result_folder / "heldout" / attack
    metadata = json.loads((full_folder / "metadata.json").read_text(encoding="utf-8"))
    expected_fit = "reduced_training_only" if scenario == "heldout" else "training_only"
    if metadata["preprocessing_fit"] != expected_fit:
        raise ValueError("Final preprocessing does not match the selected scenario.")
    if scenario == "heldout":
        names = np.load(full_folder / "train_attack_labels.npy", allow_pickle=False)
        if (names == attack).any() or metadata["held_out_attack"] != attack:
            raise ValueError("Held-out attack is present in final training data.")
    final = {name: np.load(full_folder / f"{name}.npy", allow_pickle=False)
             for name in ["X_train", "X_test", "y_train", "y_test"]}
    return train, valid, y[train_rows], y[valid_rows], final, result_folder, {
        "validation_fraction": 0.2, "validation_split_seed": VALIDATION_SEED,
        "validation_train_source_rows": raw.index.to_numpy()[train_rows].tolist(),
        "validation_source_rows": raw.index.to_numpy()[valid_rows].tolist(),
        "preprocessing_fit": "internal_training_fold_only",
        "held_out_attack_excluded": attack if scenario == "heldout" else None,
    }


def run_scenario(scenario, factory, config):
    attack = config["held_out"]["attack_type"]
    output = ROOT / "results/flyhash" / scenario
    output.mkdir(parents=True, exist_ok=True)
    train, valid, y_train, y_valid, final, baseline_folder, protocol = scenario_data(scenario, attack)
    write_json(output / "validation_protocol.json", protocol)
    ratios = config["flyhash"]["expansion_ratios"]
    sparsities = config["flyhash"]["sparsity_fractions"]
    seeds = config["seeds"]
    grid = []
    candidates = []
    for ratio in ratios:
        for sparsity in sparsities:
            records = []
            for seed in seeds:
                print(f"{scenario}: VALIDATION expansion={ratio} sparsity={sparsity} seed={seed}", flush=True)
                record, _, _ = fit_trial(factory, train, valid, y_train, y_valid,
                                         ratio, sparsity, seed, config["perceptron"])
                record["evaluation"] = "training_validation_only"
                write_json(output / f"validation_r{ratio}_s{sparsity}_seed{seed}.json", record)
                grid.append(record)
                records.append(record)
            candidates.append({
                "expansion_ratio": ratio, "sparsity": sparsity,
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in records])),
                "balanced_accuracy_sd": float(np.std([r["balanced_accuracy"] for r in records], ddof=1)),
                "false_positive_rate_mean": float(np.mean([r["false_positive_rate"] for r in records])),
            })
    best = min(candidates, key=lambda row: (-row["balanced_accuracy_mean"], row["false_positive_rate_mean"], row["expansion_ratio"], row["sparsity"]))
    selection = {"scenario": scenario, "selected": best, "candidates": candidates,
                 "objective": "Maximum mean validation balanced accuracy across seeds; ties use lower FPR, expansion and sparsity",
                 "selection_used_final_test": False, "validation_split_seed": VALIDATION_SEED}
    write_json(output / "selection.json", selection)
    with (output / "validation_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["expansion_ratio", "sparsity", "seed", "accuracy", "precision", "recall", "false_positive_rate", "balanced_accuracy", "roc_auc"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(grid)

    matrix = np.asarray([[next(row["balanced_accuracy_mean"] for row in candidates if row["expansion_ratio"] == ratio and row["sparsity"] == sparsity)
                          for sparsity in sparsities] for ratio in ratios])
    figure, axis = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    artist = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(ratios)):
        for j in range(len(sparsities)):
            axis.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center", color="white" if matrix[i, j] > 0.6 else "black")
    axis.set(xticks=range(len(sparsities)), yticks=range(len(ratios)), xlabel="WTA sparsity", ylabel="Expansion ratio", title=f"{scenario}: mean validation balanced accuracy")
    axis.set_xticklabels([f"{s:.0%}" for s in sparsities])
    axis.set_yticklabels([f"{r}x" for r in ratios])
    figure.colorbar(artist, ax=axis)
    figure.savefig(output / "validation_grid.png", dpi=180)
    plt.close(figure)

    final_records = []
    baseline_records = []
    first_scores = None
    for seed in seeds:
        print(f"{scenario}: FINAL TEST seed={seed}; settings selected on validation", flush=True)
        record, predictions, scores = fit_trial(factory, final["X_train"], final["X_test"], final["y_train"], final["y_test"],
                                               best["expansion_ratio"], best["sparsity"], seed, config["perceptron"])
        record.update({"model": "flyhash_perceptron", "evaluation": scenario,
                       "held_out_attack": attack if scenario == "heldout" else None,
                       "selection_used_final_test": False})
        write_json(output / f"flyhash_seed{seed}.json", record)
        np.savez_compressed(output / f"flyhash_seed{seed}_predictions.npz", y_true=final["y_test"], y_pred=predictions, decision_scores=scores)
        with np.load(baseline_folder / f"raw_baseline_seed{seed}_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["y_true"], final["y_test"]):
                raise ValueError("Baseline and FlyHash test rows do not match.")
            baseline_scores = saved["decision_scores"].copy()
        baseline = json.loads((baseline_folder / f"raw_baseline_seed{seed}.json").read_text(encoding="utf-8"))
        baseline["roc_auc"] = float(roc_auc_score(final["y_test"], baseline_scores))
        if baseline["learning_rate"] != config["perceptron"]["learning_rate"] or baseline["maximum_epochs"] != config["perceptron"]["epochs"]:
            raise ValueError("Baseline classifier settings do not match FlyHash settings.")
        final_records.append(record)
        baseline_records.append(baseline)
        if seed == seeds[0]:
            first_scores = (baseline_scores, scores)
    rows = []
    for model, records in [("raw_feature_perceptron", baseline_records), ("flyhash_perceptron", final_records)]:
        row = {"scenario": scenario, "model": model, "runs": len(records)}
        for metric in ["accuracy", "precision", "recall", "false_positive_rate", "roc_auc"]:
            values = np.asarray([r[metric] for r in records if r[metric] is not None], dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[f"{metric}_n_defined"] = len(values)
        rows.append(row)
    write_json(output / "comparison.json", {"selection": selection, "rows": rows})
    figure, axis = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    for label, scores in zip(["Raw features", "FlyHash"], first_scores):
        fpr, tpr, _ = roc_curve(final["y_test"], scores)
        axis.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(final['y_test'], scores):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="Attack recall", title=f"{scenario}: fixed seed {seeds[0]}")
    axis.legend()
    figure.savefig(output / "comparison_roc.png", dpi=180)
    plt.close(figure)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["standard", "heldout", "all"], default="all")
    args = parser.parse_args()
    try:
        factory = load_encoder_class()
    except MissingEncoderError as exc:
        parser.exit(2, str(exc) + "\n")
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    if len(set(config["seeds"])) < 3:
        parser.error("At least three distinct seeds are required.")
    rows = []
    for scenario in (["standard", "heldout"] if args.scenario == "all" else [args.scenario]):
        rows.extend(run_scenario(scenario, factory, config))
    folder = ROOT / "results/flyhash"
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "src/flyhash_encoder.py").open("rb") as stream:
        encoder_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(folder / "encoder_provenance.json", {"encoder_sha256": encoder_hash, "scenarios": args.scenario})
    print("FlyHash comparisons completed:", folder)


if __name__ == "__main__":
    main()
