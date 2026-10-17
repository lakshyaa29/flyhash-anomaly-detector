# FlyHash Network Anomaly Detector

Models and benchmarking for the team sprint by **Siddhesh, Hansika and Lakshya**. Mentor: Ashutosh Gopale.

**Status:** all work that can run without Siddhesh's FlyHash encoder is complete. The repository contains a NumPy perceptron, reproducible preprocessing, six real NSL-KDD baseline runs, tests, tables, plots, screenshots and a three-page report. The encoder and measured FlyHash comparison remain pending. The presentation is deferred.

## Recorded results

Mean ± sample SD across seeds **42, 43 and 44**, with a fixed test split in each scenario. Rates are percentages; their SDs are percentage points. AUC is unitless.

| Raw-feature perceptron | Accuracy | Precision | Attack recall | False-positive rate | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Standard NSL-KDD test | 76.75 ± 0.99 | 91.75 ± 0.26 | 65.00 ± 2.02 | 7.73 ± 0.45 | 0.780 ± 0.023 |
| Held-out neptune | 90.08 ± 4.16 | 83.85 ± 2.13 | 85.69 ± 12.91 | 7.81 ± 0.12 | 0.903 ± 0.076 |

The held-out test has a different attack mix and class balance; its accuracy is not a model improvement over the standard test. All six models reached the 100-epoch cap with nonzero final updates. Convergence is not demonstrated. Seed SD is not a confidence interval.

![ROC curves for three seeds in each scenario](results/plots/baseline_roc.png)

Read the [three-page report](report/baseline_report.pdf), [editable report source](report/report_draft.md), [screenshots and execution evidence](report/development_evidence.md), or [per-metric summary](results/baseline_metric_summary.csv).

## Quick start on Windows

Use Python 3.14.0, matching the recorded benchmark. Open PowerShell in this repository folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_recorded_results.py
```

The tests and recorded-result checks run without the dataset or encoder. On Linux/macOS use `python3.14 -m venv .venv` and `.venv/bin/python` in place of the Windows interpreter path.

Reproduce **all completed baseline work** with one command:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --download
```

This downloads the public dataset, runs the synthetic check, exports both data scenarios, trains six baselines, verifies predictions, regenerates plots and rebuilds the Markdown report. Training takes several minutes and depends on the machine. Existing per-seed outputs are replaced by the new run. Use `run_pipeline.py` without `--download` when the original data files already exist.

To regenerate the PDF as well:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe build_pdf_report.py
```

Or install the report dependencies first and pass `--pdf` to `run_pipeline.py`.

## Data and preprocessing

Source: [NSL-KDD Kaggle mirror by Hassan](https://www.kaggle.com/datasets/hassan06/nslkdd). `fetch_data.py` fetches its public archive and selects only the top-level `KDDTrain+.txt` and `KDDTest+.txt` members. Download source, row counts and SHA-256 checksums are recorded in [data/nsl_kdd_source.json](data/nsl_kdd_source.json). If automated download is unavailable, download from that page and put the two complete text files in `data/`.

Original partitions contain **125,973 training** and **22,544 test** records, each with 43 fields. The label and difficulty columns are excluded from the features. Labels are `0 = normal`, `1 = attack`.

Following Hansika's design, numeric features use median imputation and standardization; protocol type, service and flag use most-frequent imputation and one-hot encoding with unknown categories ignored. Fit only on training records. The same fitted transformer handles test records. Saved matrices are dense float32 arrays with 122 features.

The exporter interface is `processed_data/{X_train,X_test,y_train,y_test}.npy` and `metadata.json`, plus original attack-label arrays. `X` has shape `(n, d)` and `y` has shape `(n,)`. The baseline consumes this interface directly.

For the held-out experiment, **neptune is an individual attack type** fixed from Hansika's example before evaluation. Remove all 41,214 neptune training rows before fitting a new preprocessor and models. The remaining training set has 84,759 records. Evaluate only the original KDDTest+ normal (9,711) and neptune (4,657) records. The resulting 14,368-row test and reduced training arrays are saved in `processed_data/heldout/neptune/`, with source row indices, attack mask and metadata. Neither test population is used to select baseline settings.

## Models, metrics and experiments

`Perceptron(learning_rate=0.01, epochs=100, random_state=42).train(X, y)` returns the fitted model; `.predict(X)` returns 0/1 labels and `.decision_function(X)` returns continuous scores. Training starts with zero weights and bias, shuffles rows per epoch using the seed, updates on misclassification, and stops after a zero-update epoch or the epoch cap. A score at least zero predicts an attack.

[experiment_config.json](experiment_config.json) fixes learning rate 0.01, 100 maximum epochs, and seeds `[42, 43, 44]`. `src/metrics.py` records accuracy, precision, recall, FPR and confusion counts. Recall is TP/(TP+FN), and FPR is FP/(FP+TN). Undefined ratios are JSON `null`; summaries include the number of defined values. ROC curves and AUC use continuous decision scores, which are not calibrated probabilities.

To run stages separately:

```powershell
.\.venv\Scripts\python.exe check_synthetic.py
.\.venv\Scripts\python.exe preprocess_data.py
.\.venv\Scripts\python.exe verify_preprocessing.py
.\.venv\Scripts\python.exe run_baselines.py
.\.venv\Scripts\python.exe prepare_heldout_data.py --attack neptune
.\.venv\Scripts\python.exe run_heldout_baselines.py --attack neptune
.\.venv\Scripts\python.exe analyze_baselines.py
.\.venv\Scripts\python.exe build_baseline_report.py
```

For one standard run, use `train_baseline.py --seed 42`. The synthetic check uses a separate easy dataset: 225 train and 75 test rows, with all 75 correctly classified. It is an implementation check, not an intrusion-detection benchmark.

Full analysis recomputes metrics from saved predictions, checks thresholds and test-label alignment, verifies held-out row membership and reduced-training scaling, and checks standard processed-file hashes. It requires locally generated processed arrays and prediction archives. `verify_recorded_results.py` separately validates the six published JSON files, CSV tables and summary without those large files.

## Integrate Siddhesh's encoder later

The actual encoder is deliberately pending. The integration harness is ready in [run_flyhash_benchmarks.py](run_flyhash_benchmarks.py); it stops clearly without generating results while the encoder is missing. See [docs/ENCODER_HANDOFF.md](docs/ENCODER_HANDOFF.md) for the exact class interface and winner-count convention.

Once the real `src/flyhash_encoder.py` is supplied and all baseline inputs are prepared:

```powershell
.\.venv\Scripts\python.exe run_flyhash_benchmarks.py --scenario all
```

The runner evaluates expansion ratios `[5, 10, 20]` and sparsities `[0.02, 0.05, 0.10]` with three seeds per setting. Each scenario uses an internal stratified 80/20 split of its permitted training data (split seed 2026), with preprocessing fitted on the internal training fold. Mean validation balanced accuracy selects the setting; ties use lower FPR, expansion ratio and sparsity. Fresh models then use the selected setting on the full corresponding training partition and compare against the matching baseline on the original test population. Held-out neptune is excluded throughout training and validation.

Expected outputs include a validation grid, a sparsity/expansion heatmap, selected settings, three final runs per scenario, a baseline/FlyHash comparison table, ROC plots and encoder provenance. These outputs are not included until the real encoder is run. Interface checks use test doubles; end-to-end FlyHash validation remains pending.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | NumPy classifier, metrics and encoder interface checks |
| `tests/` | 11 tests for model behavior, metrics, preprocessing isolation and encoder contract |
| `.github/workflows/checks.yml` | Unit tests and recorded-result checks on push/pull request |
| `data/` | Dataset provenance and download instructions |
| `results/` | Six real run records, summary tables, plots and captured logs |
| `report/` | Three-page PDF, Markdown source, uploaded screenshots and execution evidence |
| `docs/ENCODER_HANDOFF.md` | Pending encoder interface and usage |
| `run_pipeline.py` | All completed baseline stages in order |
| `package_github.py` | Builds a clean shareable folder and ZIP |
| `UPLOAD_TO_GITHUB.md` | Upload instructions |

Raw data, processed arrays, prediction archives, virtual environments, caches and private working notes are excluded from the upload package and by `.gitignore`. Reproduce them with the pipeline. The original submitted synthetic screenshots and original JSON evidence are retained unchanged; their older package versions differ from the current benchmark environment recorded in each real-run JSON.

The clean export converts text line endings to LF and replaces this laptop's absolute project path in logs/JSON with `.`. Experiment hashes in the summary preserve original execution provenance; `PACKAGE_MANIFEST.json` separately lists the exact exported-file checksums.

## Team contributions and current limits

Hansika supplied the preprocessing design and reference experiment scripts. Lakshya's component contains the perceptron, metric evaluation, repeated baseline experiments, plots and report. Siddhesh owns the FlyHash encoder. The supplied `phase4.py` was used as a reference and was not run as the benchmark.

The baseline and interface tests have been run locally. The GitHub Actions workflow will run after upload; hosted CI has not yet been executed. Workflow action examples follow the official [checkout](https://github.com/actions/checkout) and [setup-python](https://github.com/actions/setup-python) documentation.

The joint FlyHash comparison and final discussion depend on the encoder. The presentation is deferred.
#   f l y h a s h - a n o m a l y - d e t e c t o r  
 