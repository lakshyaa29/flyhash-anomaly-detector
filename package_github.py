"""Export only the public project deliverables; never deletes or overwrites a package."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
TEXT = {".py", ".md", ".json", ".csv", ".txt", ".yml", ".yaml"}
ROOT_FILES = [
    ".gitignore", ".gitattributes", "README.md", "UPLOAD_TO_GITHUB.md",
    "requirements.txt", "requirements-report.txt", "experiment_config.json",
    "analyze_baselines.py", "build_baseline_report.py", "build_pdf_report.py",
    "check_setup.py", "check_synthetic.py", "fetch_data.py", "package_github.py",
    "prepare_heldout_data.py", "preprocess_data.py", "run_baselines.py",
    "run_flyhash_benchmarks.py", "run_heldout_baselines.py", "run_pipeline.py",
    "train_baseline.py", "verify_preprocessing.py", "verify_recorded_results.py",
]


def selected_files():
    files = [ROOT / name for name in ROOT_FILES]
    for directory in ["src", "tests", ".github", "docs", "notebooks"]:
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix in TEXT)
    files.extend([ROOT / "data/README.md", ROOT / "data/nsl_kdd_source.json"])
    files.extend(path for path in (ROOT / "results").rglob("*")
                 if path.is_file() and path.suffix in {".json", ".csv", ".txt", ".png"})
    files.extend([ROOT / "report/report_draft.md", ROOT / "report/development_evidence.md",
                  ROOT / "report/baseline_report.pdf", ROOT / "report/evidence/synthetic_results_original.json"])
    files.extend((ROOT / "report/figures").glob("*.png"))
    files = sorted(set(files))
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Build required artifacts first: {missing}")
    return files


def portable_text(path):
    text = path.read_text(encoding="utf-8-sig")
    # Preserve original evidence JSON exactly; its screenshots record an earlier environment.
    if path.relative_to(ROOT).as_posix() == "report/evidence/synthetic_results_original.json":
        return path.read_bytes()
    if path.suffix == ".json":
        def normalize(value):
            if isinstance(value, str):
                return value.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value
        text = json.dumps(normalize(json.loads(text)), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    elif path.suffix in {".md", ".txt", ".csv"}:
        text = text.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "github-ready")
    args = parser.parse_args()
    sources = selected_files()
    parent = args.output_dir.resolve()
    folder = parent / "flyhash-anomaly-detector"
    archive = parent / "flyhash-anomaly-detector-github.zip"
    if folder.exists() or archive.exists():
        parser.error(f"Output already exists in {parent}. Choose a new --output-dir; nothing was overwritten.")
    folder.mkdir(parents=True)
    manifest = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        output = folder / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        data = portable_text(source) if source.suffix in TEXT or source.name.startswith(".git") else source.read_bytes()
        output.write_bytes(data)
        manifest[relative.as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Baseline work complete; actual FlyHash encoder and comparison pending; presentation deferred",
        "excluded": ["raw datasets", "download archive", "processed arrays", "prediction archives",
                     "virtual environment", "caches", "private notes and report history", "presentation"],
        "text_normalization": "UTF-8 LF; project absolute paths in logs/JSON/Markdown replaced with .; original screenshot evidence retained",
        "provenance_note": "Experiment hashes describe original execution files; hashes below describe this export. This manifest excludes itself.",
        "file_count_without_manifest": len(manifest),
        "files": manifest,
    }
    (folder / "PACKAGE_MANIFEST.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with ZipFile(archive, "x", ZIP_DEFLATED) as zip_file:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zip_file.write(path, path.relative_to(parent).as_posix())
    with ZipFile(archive) as zip_file:
        if zip_file.testzip() is not None:
            raise ValueError("Archive integrity check failed.")
        if len(zip_file.infolist()) != len(manifest) + 1:
            raise ValueError("Archive file count mismatch.")
    print(f"Exported {len(manifest) + 1} files to: {folder}")
    print(f"Verified ZIP ({archive.stat().st_size:,} bytes): {archive}")


if __name__ == "__main__":
    main()
