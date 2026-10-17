"""Reproduce all completed benchmark work using the current interpreter."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Fetch the public Kaggle archive first.")
    parser.add_argument("--pdf", action="store_true", help="Also build the PDF; install requirements-report.txt first.")
    args = parser.parse_args()
    steps = ["fetch_data.py"] if args.download else []
    steps += ["check_synthetic.py", "preprocess_data.py", "verify_preprocessing.py", "run_baselines.py",
              "prepare_heldout_data.py", "run_heldout_baselines.py", "analyze_baselines.py", "build_baseline_report.py"]
    if args.pdf:
        steps.append("build_pdf_report.py")
    logs = {
        "check_synthetic.py": "synthetic_check_log.txt",
        "preprocess_data.py": "preprocessing_run.txt",
        "verify_preprocessing.py": "preprocessing_validation_log.txt",
        "run_baselines.py": "baseline_runs_log.txt",
        "prepare_heldout_data.py": "heldout_preprocessing_log.txt",
        "run_heldout_baselines.py": "heldout_baseline_runs_log.txt",
        "analyze_baselines.py": "baseline_analysis_log.txt",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    for name in steps:
        print(f"\n=== {name} ===", flush=True)
        log_path = ROOT / "results" / logs.get(name, name.replace(".py", "_log.txt"))
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen([sys.executable, "-u", str(ROOT / name)], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
    print("\nCompleted all baseline work. FlyHash requires Siddhesh's encoder.", flush=True)


if __name__ == "__main__":
    main()
