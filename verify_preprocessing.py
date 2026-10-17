"""Verify saved standard-split arrays without fitting on test data."""

import hashlib
import json
from pathlib import Path

import numpy as np

from preprocess_data import CATEGORICAL, NUMERIC, load_kdd


ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    root = Path(root)
    folder = root / "processed_data"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata["label_mapping"] != {"0": "normal", "1": "attack"}:
        raise ValueError("Unexpected binary label mapping.")
    if metadata["preprocessing_fit"] != "training_only":
        raise ValueError("Standard preprocessing must be fitted on training data only.")
    checks = {}
    for split, source in [("train", "KDDTrain+.txt"), ("test", "KDDTest+.txt")]:
        raw = load_kdd(root / "data" / source)
        X = np.load(folder / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(folder / f"y_{split}.npy", allow_pickle=False)
        labels = np.load(folder / f"{split}_attack_labels.npy", allow_pickle=False)
        if X.shape != (len(raw), len(metadata["feature_names"])) or X.dtype != np.float32:
            raise ValueError(f"Unexpected {split} feature shape or dtype.")
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite {split} feature values.")
        if not np.array_equal(labels, raw["label"].to_numpy(dtype=str)):
            raise ValueError(f"{split} source labels or row order do not match.")
        if not np.array_equal(y, (labels != "normal").astype(np.int64)):
            raise ValueError(f"{split} binary labels do not match attack names.")
        if not np.isin(X[:, len(NUMERIC):], [0, 1]).all():
            raise ValueError("One-hot columns must be binary.")
        groups = {}
        for name in CATEGORICAL:
            indices = [i for i, field in enumerate(metadata["feature_names"]) if field.startswith(f"categorical__{name}_")]
            sums = X[:, indices].sum(axis=1)
            if not np.isin(sums, [0, 1]).all() or (split == "train" and not (sums == 1).all()):
                raise ValueError(f"Invalid {split} category group: {name}.")
            groups[name] = {"encoded_features": len(indices), "unknown_category_rows": int((sums == 0).sum())}
        checks[split] = {
            "shape": list(X.shape), "dtype": str(X.dtype),
            "normal_samples": int((y == 0).sum()), "attack_samples": int((y == 1).sum()),
            "label_alignment": True, "features_finite": True, "categorical_groups": groups,
        }
        if split == "train":
            means = X[:, :len(NUMERIC)].mean(axis=0, dtype=np.float64)
            variances = X[:, :len(NUMERIC)].var(axis=0, dtype=np.float64)
            if np.max(np.abs(means)) >= 1e-5 or np.max(np.minimum(np.abs(variances), np.abs(variances - 1))) >= 1e-5:
                raise ValueError("Training numeric columns are not standardized.")
            checks[split]["maximum_absolute_numeric_mean"] = float(np.max(np.abs(means)))
            checks[split]["numeric_variances_zero_or_one"] = True
    hashes = {}
    for path in sorted(folder.glob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"status": "passed", "checks": checks, "sha256": hashes}
    output = root / "results" / "preprocessing_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PREPROCESSING VERIFICATION PASSED:", output)
    return result


if __name__ == "__main__":
    verify()
