import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_data import load_kdd, make_preprocessor


ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(attack):
    if not attack or not all(character.isalnum() or character == "_" for character in attack):
        raise ValueError("Use an attack label consisting of letters, digits or underscores.")

    train_path = ROOT / "data" / "KDDTrain+.txt"
    test_path = ROOT / "data" / "KDDTest+.txt"
    train = load_kdd(train_path)
    test = load_kdd(test_path)
    removed_mask = train["label"].eq(attack)
    selected_test = test["label"].isin(["normal", attack])

    if not removed_mask.any() or not test["label"].eq(attack).any():
        raise ValueError(f"{attack!r} must occur in both original source files.")

    reduced = train.loc[~removed_mask].copy()
    evaluation = test.loc[selected_test].copy()
    if reduced["label"].eq(attack).any():
        raise AssertionError("Held-out attack is still present in reduced training data.")

    y_train = reduced["label"].ne("normal").to_numpy(dtype=np.int64)
    y_test = evaluation["label"].ne("normal").to_numpy(dtype=np.int64)
    attack_mask = evaluation["label"].eq(attack).to_numpy(dtype=bool)
    if not np.array_equal(y_test, attack_mask.astype(np.int64)):
        raise AssertionError("Evaluation contains an unexpected attack type.")

    preprocessor = make_preprocessor()
    print(f"Fitting preprocessing with {attack} excluded...", flush=True)
    X_train = np.asarray(
        preprocessor.fit_transform(reduced.drop(columns=["label", "difficulty"])),
        dtype=np.float32,
    )
    X_test = np.asarray(
        preprocessor.transform(evaluation.drop(columns=["label", "difficulty"])),
        dtype=np.float32,
    )
    if X_train.shape[1] != X_test.shape[1] or not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise AssertionError("Invalid processed feature matrices.")

    output = ROOT / "processed_data" / "heldout" / attack
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_attack_labels": reduced["label"].to_numpy(dtype=str),
        "test_attack_labels": evaluation["label"].to_numpy(dtype=str),
        "heldout_attack_mask": attack_mask,
        "train_source_rows": reduced.index.to_numpy(dtype=np.int64),
        "test_source_rows": evaluation.index.to_numpy(dtype=np.int64),
    }
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)

    pd.DataFrame({
        "source_file": "KDDTest+.txt",
        "source_row_zero_based": evaluation.index.to_numpy(),
        "original_attack_name": evaluation["label"].to_numpy(),
        "binary_label": y_test,
        "is_heldout_attack": attack_mask,
    }).to_csv(output / "sample_info.csv", index=False)

    metadata = {
        "dataset": "NSL-KDD",
        "evaluation": "heldout_attack",
        "held_out_attack": attack,
        "label_mapping": {"0": "normal", "1": "attack"},
        "training_samples_removed": int(removed_mask.sum()),
        "training_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "training_label_counts": {
            "normal": int((y_train == 0).sum()),
            "attack": int((y_train == 1).sum()),
        },
        "test_label_counts": {
            "normal": int((y_test == 0).sum()),
            "heldout_attack": int(attack_mask.sum()),
        },
        "preprocessing_fit": "reduced_training_only",
        "excluded_before_preprocessing_fit": True,
        "heldout_attack_in_training": False,
        "test_source": "Original KDDTest+.txt rows labelled normal or the selected attack",
        "rows_preserved_in_source_order": True,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "raw_sha256": {train_path.name: sha256(train_path), test_path.name: sha256(test_path)},
        "processed_sha256": {f"{name}.npy": sha256(output / f"{name}.npy") for name in arrays},
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "package_versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn"]},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("HELD-OUT PREPROCESSING COMPLETED")
    print("Attack type:", attack)
    print("Training rows removed:", metadata["training_samples_removed"])
    print("Reduced training shape:", X_train.shape)
    print("Held-out evaluation shape:", X_test.shape)
    print("Evaluation classes:", metadata["test_label_counts"])
    print("Saved to:", output)
    return metadata


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default=config["held_out"]["attack_type"])
    args = parser.parse_args()
    prepare(args.attack)


if __name__ == "__main__":
    main()
