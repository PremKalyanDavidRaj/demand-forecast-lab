"""Download and record the original UCI Bike Sharing dataset."""

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike%2Bsharing%2Bdataset.zip"
)
FILES = ("hour.csv", "day.csv", "Readme.txt")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Bike Sharing dataset...")
    with urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()

    contents = {}
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        for filename in FILES:
            matches = [
                name for name in archive.namelist()
                if Path(name).name == filename
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {filename} in the archive."
                )
            contents[filename] = archive.read(matches[0])

    # Check all existing files before writing any replacements.
    for filename, content in contents.items():
        path = RAW / filename
        if path.exists() and path.read_bytes() != content:
            raise ValueError(
                f"{path} differs from the download. "
                "Existing file preserved; investigate before replacing it."
            )

    file_records = {}
    for filename, content in contents.items():
        path = RAW / filename
        path.write_bytes(content)
        file_records[filename] = {
            "bytes": len(content),
            "sha256": sha256(content),
        }
        print(f"Saved {filename}: {len(content):,} bytes")

    manifest = {
        "dataset": "UCI Bike Sharing",
        "source_url": URL,
        "dataset_doi": "https://doi.org/10.24432/C5W894",
        "license": "CC BY 4.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": sha256(archive_bytes),
        "files": file_records,
        "note": (
            "Original file bytes preserved. Hashes record this download; "
            "they are not independently verified publisher checksums."
        ),
    }

    destination = REPORTS / "data_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Manifest saved to reports/data_manifest.json")


if __name__ == "__main__":
    main()
