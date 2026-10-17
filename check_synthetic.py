import json
import sys
from pathlib import Path
from importlib.metadata import version

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from src.perceptron import Perceptron
from src.metrics import evaluate_binary


SEED = 42

X, y = make_blobs(
    n_samples=300,
    centers=[(-2, -2), (2, 2)],
    cluster_std=0.6,
    random_state=SEED,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y,
)

model = Perceptron(
    learning_rate=0.01,
    epochs=100,
    random_state=SEED,
)

model.train(X_train, y_train)
predictions = model.predict(X_test)
results = evaluate_binary(y_test, predictions)


def format_metric(value):
    return f"{value:.4f}" if value is not None else "undefined"


print("SYNTHETIC PERCEPTRON CHECK")
print("Random seed:", SEED)
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Epochs completed:", len(model.mistakes_per_epoch))
print("Updates per epoch:", model.mistakes_per_epoch)
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

results.update({
    "dataset": "synthetic_blobs",
    "seed": SEED,
    "training_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "learning_rate": model.learning_rate,
    "maximum_epochs": model.epochs,
    "epochs_completed": len(model.mistakes_per_epoch),
    "updates_per_epoch": model.mistakes_per_epoch,
    "python_version": sys.version.split()[0],
    "package_versions": {
        package: version(package)
        for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]
    },
})

output_folder = Path(__file__).resolve().parent / "results"
output_folder.mkdir(exist_ok=True)
output_file = output_folder / "synthetic_results.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, allow_nan=False)

print("Results saved to:", output_file)