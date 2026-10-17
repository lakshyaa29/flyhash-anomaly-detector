import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
RESULTS = ROOT / "results"


def percent(entry, name):
    stats = entry["metrics"][name]
    return f"{100 * stats['mean']:.2f} ± {100 * stats['sample_sd']:.2f}"


def auc(entry):
    stats = entry["metrics"]["roc_auc"]
    return f"{stats['mean']:.3f} ± {stats['sample_sd']:.3f}"


def read_log(name):
    return (RESULTS / name).read_text(encoding="utf-8-sig").strip()


def main():
    summary = json.loads((RESULTS / "baseline_summary.json").read_text(encoding="utf-8"))
    standard = summary["scenarios"]["Standard NSL-KDD test"]
    heldout = summary["scenarios"]["Held-out neptune"]
    current_run = json.loads((RESULTS / "raw_baseline_seed42.json").read_text(encoding="utf-8"))
    heldout_runs = [json.loads((RESULTS / "heldout" / "neptune" / f"raw_baseline_seed{seed}.json").read_text(encoding="utf-8")) for seed in summary["seeds"]]
    low_recall = min(run["recall"] for run in heldout_runs)
    high_recall = max(run["recall"] for run in heldout_runs)
    versions = current_run["package_versions"]
    history = REPORT / "history"
    history.mkdir(exist_ok=True)
    previous = history / "report_before_baseline.md"
    if (REPORT / "report_draft.md").is_file() and not previous.exists():
        shutil.copy2(REPORT / "report_draft.md", previous)

    text = f"""# FlyHash Network Anomaly Detector

Team: Siddhesh, Hansika and Lakshya  
Mentor: Ashutosh Gopale

The raw-feature perceptron baseline is complete across three fixed seeds. On the standard NSL-KDD test, mean accuracy is {100 * standard['metrics']['accuracy']['mean']:.2f}%, attack recall is {100 * standard['metrics']['recall']['mean']:.2f}% and false-positive rate is {100 * standard['metrics']['false_positive_rate']['mean']:.2f}%. A separately retrained model also evaluates an attack type excluded from training. The FlyHash comparison remains pending encoder integration.

## Data and approach

The [NSL-KDD files from Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) contain 125,973 training and 22,544 test records. Original partitions are preserved. The standard test contains 9,711 normal and 12,833 attack records. Labels map normal traffic to 0 and attacks to 1. The label and difficulty columns are excluded from model features.

Preprocessing follows Hansika's pipeline: median imputation and standardization for 38 numeric features, plus most-frequent imputation and one-hot encoding of protocol type, service and flag. Fitting uses training data only; test data use the same fitted transformation. The resulting dense float32 inputs have 122 features.

The NumPy perceptron starts with zero weights and bias, shuffles training samples by seed, and updates its weights after misclassification. Fixed settings are learning rate 0.01, 100 maximum epochs and seeds 42, 43 and 44. A score at least zero predicts attack. A separate synthetic check correctly classified 75 test samples after training on 225 samples; those easy synthetic results establish basic implementation behavior.

## Held out attack protocol

The attack type `neptune` was chosen from Hansika's supplied examples before this evaluation. All 41,214 training records of that type are removed before fitting a new preprocessor. Each seed trains a fresh model on the remaining 84,759 records. Evaluation uses 4,657 `neptune` and 9,711 normal records from the original test file, totaling 14,368 records. The held-out attack is absent from training and preprocessing fitting. Model settings are fixed without tuning on this evaluation.

## Results

Values are mean ± sample SD across three training seeds on a fixed test split. Accuracy, precision, recall and FPR are percentages; their SDs are percentage points. ROC AUC is unitless. These SDs describe seed variation.

| Raw feature evaluation | Accuracy | Precision | Recall | FPR | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Standard test | {percent(standard, 'accuracy')} | {percent(standard, 'precision')} | {percent(standard, 'recall')} | {percent(standard, 'false_positive_rate')} | {auc(standard)} |
| Held out neptune | {percent(heldout, 'accuracy')} | {percent(heldout, 'precision')} | {percent(heldout, 'recall')} | {percent(heldout, 'false_positive_rate')} | {auc(heldout)} |

![Baseline ROC curves](figures/baseline_roc.png)

Figure 1. ROC curves for all three seeds, computed from continuous decision scores. Dots mark the fixed score-zero operating points. Each panel uses its stated training and test protocol.

![Baseline confusion matrices](figures/baseline_confusion_matrices.png)

Figure 2. Confusion matrices for the fixed seed 42. Annotations show counts and percentages within each actual class. Seed 42 was used consistently for illustration rather than selected by its performance.

## Interpretation and limitations

The standard baseline misses about {100 * (1 - standard['metrics']['recall']['mean']):.1f}% of attacks and flags about {100 * standard['metrics']['false_positive_rate']['mean']:.1f}% of normal traffic. Held-out `neptune` recall ranges from {100 * low_recall:.2f}% to {100 * high_recall:.2f}% across seeds, showing sensitivity to training order. The restricted held-out test has a different attack mix and class balance, so its accuracy should be interpreted within that scenario. One held-out attack and one source split limit the scope of the generalization claim.

All six runs reached the 100-epoch limit with nonzero updates in their final epoch. Convergence has not been demonstrated. No threshold or hyperparameter tuning was performed on either evaluation set.

## Contributions and remaining work

Hansika supplied the data preprocessing pipeline and reference experiment scripts. Lakshya's component comprises the NumPy perceptron, metric evaluator, repeated baseline experiments, tables, plots and report results. Siddhesh owns the FlyHash encoder, whose integration is pending.

The prepared benchmark runner will compare that encoder plus the same classifier against these baselines, using the same inputs and seeds. Planned expansion ratios are 5×, 10× and 20×, and winner-take-all sparsities are 2%, 5% and 10%, with three runs per setting. Parameter selection uses a training-only validation fold and preserves the held-out exclusion. The final joint report will incorporate that comparison. The presentation is deferred.

## Reproducibility

The benchmark used Python {current_run['python_version']}, NumPy {versions['numpy']}, pandas {versions['pandas']}, scikit-learn {versions['scikit-learn']} and Matplotlib {versions['matplotlib']}. Run commands are in the [README](../README.md). Per-seed metrics and prediction scores are retained under results/. [The summary](../results/baseline_summary.json) records means, SDs, source/code hashes and output checks. Those checks recomputed metrics, verified prediction thresholds and test labels, and confirmed held-out source-row membership and reduced-training scaling.

Uploaded screenshots, captured execution logs and the training-update plot are preserved in the [development evidence](development_evidence.md). The current [three-page PDF report](baseline_report.pdf) covers the completed baseline work; its FlyHash comparison remains pending.
"""
    (REPORT / "report_draft.md").write_text(text, encoding="utf-8")

    evidence = f"""# Development Evidence for the FlyHash Sprint

This supporting record preserves the submitted screenshots and actual execution logs while the main report presents the measured benchmark results.

## Synthetic classifier check

![Submitted synthetic test output](figures/01_synthetic_perceptron_check.png)

Submitted output with seed 42, 225 training and 75 test samples. Accuracy, precision and recall are 1.0000; FPR is 0.0000; the confusion matrix is [[38, 0], [0, 37]].

## Saved synthetic result

![Submitted saved JSON result](figures/02_synthetic_results_json.png)

The submitted screenshot agrees with the saved synthetic result file.

## Submitted synthetic environment

![Submitted synthetic environment versions](figures/03_environment_versions.png)

This earlier submitted run records Python 3.14.0, NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.1 and Matplotlib 3.11.1. Current project benchmark versions are recorded separately in every benchmark JSON and the main report.

## Standard preprocessing execution

The following is captured stdout from the actual execution, not a screenshot.

```text
{read_log('preprocessing_run.txt')}
```

## Standard baseline executions

```text
{read_log('baseline_runs_log.txt')}
```

## Held out preprocessing execution

```text
{read_log('heldout_preprocessing_log.txt')}
```

## Held out baseline executions

```text
{read_log('heldout_baseline_runs_log.txt')}
```

## Online training updates

![Weight updates during training](figures/baseline_training_updates.png)

Each curve counts online weight updates while visiting shuffled training samples during an epoch. This is not a test-error or accuracy curve. All six runs still made updates at epoch 100.

## Output verification and summary

```text
{read_log('baseline_analysis_log.txt')}
```

The analysis checked that saved predictions, scores and metrics agree; that the same test labels are used across seeds; that held-out samples come from the intended original test rows; and that preprocessing fitting uses the reduced training partition. Raw data provenance, processed-file hashes, row indices and label masks remain available in the project.
"""
    (REPORT / "development_evidence.md").write_text(evidence, encoding="utf-8")
    print("Updated:", REPORT / "report_draft.md")
    print("Preserved screenshots and execution evidence:", REPORT / "development_evidence.md")


if __name__ == "__main__":
    main()
