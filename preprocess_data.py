import json
import sys
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "processed_data"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

NUMERIC = [
    column
    for column in COLUMNS[:-2]
    if column not in CATEGORICAL
]


def load_kdd(path):
    frame = pd.read_csv(path, header=None)

    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected 43 columns, got {frame.shape[1]}."
        )

    frame.columns = COLUMNS

    if frame["label"].isna().any():
        raise ValueError(f"{path.name}: missing attack labels.")

    frame["label"] = (
        frame["label"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
    )

    if frame["label"].eq("").any():
        raise ValueError(f"{path.name}: empty attack labels.")

    return frame


def make_preprocessor():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer(
        [
            ("numerical", numerical_pipeline, NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL),
        ],
        sparse_threshold=0.0,
    )


def main():
    print("Project folder:", ROOT)
    print("Loading NSL-KDD datasets...", flush=True)

    train = load_kdd(DATA_DIR / "KDDTrain+.txt")
    test = load_kdd(DATA_DIR / "KDDTest+.txt")

    print("Raw training shape:", train.shape)
    print("Raw test shape:", test.shape)

    X_train_raw = train.drop(columns=["label", "difficulty"])
    X_test_raw = test.drop(columns=["label", "difficulty"])

    y_train = (train["label"] != "normal").to_numpy(dtype=np.int64)
    y_test = (test["label"] != "normal").to_numpy(dtype=np.int64)

    print("Preprocessing...", flush=True)

    preprocessor = make_preprocessor()

    X_train = np.asarray(
        preprocessor.fit_transform(X_train_raw),
        dtype=np.float32,
    )

    X_test = np.asarray(
        preprocessor.transform(X_test_raw),
        dtype=np.float32,
    )

    if not np.isfinite(X_train).all():
        raise ValueError("Processed training features contain NaN or infinity.")

    if not np.isfinite(X_test).all():
        raise ValueError("Processed test features contain NaN or infinity.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test feature counts do not match.")

    train_counts = {
        "normal": int((y_train == 0).sum()),
        "attack": int((y_train == 1).sum()),
    }

    test_counts = {
        "normal": int((y_test == 0).sum()),
        "attack": int((y_test == 1).sum()),
    }

    arrays = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_attack_labels": train["label"].to_numpy(dtype=str),
        "test_attack_labels": test["label"].to_numpy(dtype=str),
    }

    metadata = {
        "dataset": "NSL-KDD",
        "evaluation": "standard_split",
        "preprocessing_source": "Adapted from Hansika's phase01.py",
        "label_mapping": {"0": "normal", "1": "attack"},
        "training_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "training_label_counts": train_counts,
        "test_label_counts": test_counts,
        "preprocessing_fit": "training_only",
        "excluded_columns": ["label", "difficulty"],
        "numeric_preprocessing": "median imputation and standardization",
        "categorical_preprocessing": "most-frequent imputation and one-hot",
        "unknown_categories": "ignore",
        "feature_dtype": str(X_train.dtype),
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "held_out_attacks": {},
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "package_versions": {
            package: version(package)
            for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
    }

    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, array in arrays.items():
        np.save(OUTPUT_DIR / f"{name}.npy", array)

    with (OUTPUT_DIR / "metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    print("\nPREPROCESSING COMPLETED")
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Training labels:", train_counts)
    print("Test labels:", test_counts)
    print("Feature dtype:", X_train.dtype)
    print("Preprocessing fitted on: training data only")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
