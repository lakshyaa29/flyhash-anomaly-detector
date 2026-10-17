from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import shutil
import urllib.request
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
ARCHIVE = DATA / "downloads" / "nslkdd.zip"
SOURCE = "https://www.kaggle.com/datasets/hassan06/nslkdd"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/hassan06/nslkdd"
FILENAMES = ("KDDTrain+.txt", "KDDTest+.txt")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path):
    labels = Counter()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for rows, row in enumerate(csv.reader(stream), start=1):
            if len(row) != 43:
                raise ValueError(f"{path.name}: row {rows} has {len(row)} columns; expected 43.")
            float(row[0])
            float(row[42])
            label = row[41].strip()
            if not label:
                raise ValueError(f"{path.name}: empty label at row {rows}.")
            labels[label] += 1
    if rows == 0 or "normal" not in labels or len(labels) < 2:
        raise ValueError(f"{path.name}: expected nonempty normal and attack records.")
    return {
        "rows": rows,
        "columns": 43,
        "sha256": digest(path),
        "normal_samples": labels["normal"],
        "attack_samples": rows - labels["normal"],
        "attack_label_counts": dict(sorted(labels.items())),
    }


def download():
    if ARCHIVE.is_file():
        print("Using cached archive:", ARCHIVE, flush=True)
        return
    request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        first = response.read(1024 * 1024)
        if not first.startswith(b"PK\x03\x04"):
            raise RuntimeError("Kaggle returned a page rather than a ZIP archive.")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        partial = ARCHIVE.with_suffix(".zip.part")
        total = len(first)
        try:
            with partial.open("wb") as output:
                output.write(first)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > 200 * 1024 * 1024:
                        raise RuntimeError("Archive exceeded the 200 MB size limit.")
                    output.write(block)
            partial.replace(ARCHIVE)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    print("Downloaded ZIP bytes:", total, flush=True)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    download()
    records = {}
    with ZipFile(ARCHIVE) as archive:
        for filename in FILENAMES:
            matches = [
                item for item in archive.infolist()
                if not item.is_dir()
                and PurePosixPath(item.filename.replace("\\", "/")).name.casefold()
                == filename.casefold()
            ]
            top_level = [item for item in matches if item.filename == filename]
            if top_level:
                matches = top_level
            if len(matches) != 1 or matches[0].file_size > 100 * 1024 * 1024:
                raise ValueError(f"Expected one reasonably sized {filename} in the archive.")
            target = DATA / filename
            partial = DATA / (filename + ".part")
            try:
                with archive.open(matches[0]) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output)
                info = validate(partial)
                if target.exists():
                    if digest(target) != info["sha256"]:
                        raise FileExistsError(f"Preserving a different existing file: {target}")
                    partial.unlink()
                else:
                    partial.rename(target)
                records[filename] = info
                print(f"Verified {filename}: {info['rows']} rows, {info['columns']} columns", flush=True)
            finally:
                partial.unlink(missing_ok=True)
    provenance = {
        "dataset": "NSL-KDD",
        "source_url": SOURCE,
        "download_url": DOWNLOAD,
        "downloaded_at_utc": (json.loads((DATA / "nsl_kdd_source.json").read_text(encoding="utf-8")).get("downloaded_at_utc") if (DATA / "nsl_kdd_source.json").is_file() else datetime.now(timezone.utc).isoformat()),
        "archive_sha256": digest(ARCHIVE),
        "files": records,
    }
    manifest = DATA / "nsl_kdd_source.json"
    manifest.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Saved dataset files to:", DATA, flush=True)
    print("Saved provenance:", manifest, flush=True)


if __name__ == "__main__":
    main()
