"""Download HAM10000 from the Harvard Dataverse with resume and verification."""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from melanoma.data.metadata import load_metadata
from melanoma.utils import ensure_dir

DATASET_DOI = "doi:10.7910/DVN/DBW86T"
DATASET_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=" + DATASET_DOI
)
DATAFILE_API = "https://dataverse.harvard.edu/api/access/datafile/{file_id}?format=original"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
DOWNLOAD_HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}


class DownloadError(RuntimeError):
    """Raised when a download, checksum, or extraction step fails."""


def fetch_file_index(session=None, timeout=60):
    """Query the Dataverse API and return {filename: {id, size, md5}}."""
    session = session or requests.Session()
    try:
        response = session.get(DATASET_API, timeout=timeout, headers=REQUEST_HEADERS)
    except requests.RequestException as exc:
        raise DownloadError(f"Dataverse API request failed: {exc}") from exc
    if response.status_code != 200:
        raise DownloadError(f"Dataverse API returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise DownloadError("Dataverse API returned invalid JSON") from exc
    files = payload.get("data", {}).get("latestVersion", {}).get("files")
    if not files:
        raise DownloadError("Dataverse response contains no files")
    index = {}
    for entry in files:
        data = entry.get("dataFile", {})
        name = data.get("filename") or data.get("name") or entry.get("label")
        file_id = data.get("id")
        if not name or file_id is None:
            continue
        md5 = data.get("md5")
        if not md5:
            checksum = data.get("checksum") or {}
            if checksum.get("type") == "MD5":
                md5 = checksum.get("value")
        index[name] = {
            "id": file_id,
            "size": data.get("filesize"),
            "md5": md5,
        }
    if not index:
        raise DownloadError("No usable file entries in Dataverse response")
    return index


def select_files(index):
    """Pick the metadata file and the two HAM10000 image zip files from the index."""
    metadata_candidates = sorted(
        name
        for name in index
        if name.lower().endswith((".csv", ".tab")) and "metadata" in name.lower()
    )
    if not metadata_candidates:
        raise DownloadError(f"No metadata file found in dataset files: {sorted(index)}")
    zips = sorted(
        name
        for name in index
        if name.lower().endswith(".zip") and "images_part" in name.lower()
    )
    if len(zips) < 2:
        raise DownloadError(f"Expected two HAM10000 image zip files, found: {zips}")
    return {"metadata": metadata_candidates[0], "images": zips}


def normalize_metadata(source, destination):
    """Convert a tab or comma separated metadata file to canonical CSV."""
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise DownloadError(f"Metadata source not found: {source}")
    frame = pd.read_csv(source, sep=None, engine="python")
    frame.to_csv(destination, index=False)
    return destination


def md5sum(path, chunk_size=1 << 20):
    """Compute the MD5 hex digest of a file."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url,
    dest,
    expected_size=None,
    expected_md5=None,
    session=None,
    chunk_size=1 << 20,
    resume=True,
    show_progress=True,
    timeout=120,
):
    """Download a URL to dest with optional resume and checksum verification.

    Returns 'cached' when dest already exists and passes checks, otherwise
    'downloaded'. MD5 is the authoritative integrity check when provided;
    expected_size is checked only when no MD5 is available (some servers
    report sizes of processed variants). Raises DownloadError on HTTP
    failures, incomplete downloads, or checksum mismatches.
    """
    dest = Path(dest)
    if dest.exists():
        size_ok = expected_size is None or dest.stat().st_size == expected_size
        md5_ok = expected_md5 is None or md5sum(dest) == expected_md5
        if size_ok and md5_ok:
            return "cached"
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = dict(DOWNLOAD_HEADERS)
    append = False
    if resume and part.exists() and part.stat().st_size > 0:
        headers["Range"] = f"bytes={part.stat().st_size}-"
        append = True

    session = session or requests.Session()
    try:
        response = session.get(url, stream=True, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise DownloadError(f"Request failed for {url}: {exc}") from exc

    if response.status_code == 416 and append:
        response.close()
        part.unlink(missing_ok=True)
        return download_file(
            url,
            dest,
            expected_size,
            expected_md5,
            session=session,
            chunk_size=chunk_size,
            resume=False,
            show_progress=show_progress,
            timeout=timeout,
        )
    if response.status_code not in (200, 206):
        raise DownloadError(f"HTTP {response.status_code} for {url}")
    if response.status_code == 200 and append:
        append = False

    initial = part.stat().st_size if (append and part.exists()) else 0
    progress = tqdm(
        total=expected_size,
        initial=initial,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        disable=not show_progress,
    )
    try:
        with part.open("ab" if append else "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    finally:
        progress.close()
        response.close()

    size = part.stat().st_size
    if expected_md5 is not None:
        actual = md5sum(part)
        if actual != expected_md5:
            part.unlink(missing_ok=True)
            raise DownloadError(f"MD5 mismatch for {dest.name}: {actual} != {expected_md5}")
    elif expected_size is not None and size != expected_size:
        raise DownloadError(f"Incomplete download for {dest.name}: {size} of {expected_size} bytes")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part.replace(dest)
    return "downloaded"


def extract_zip(zip_path, extract_dir, overwrite=False):
    """Safely extract a zip archive, rejecting path traversal members.

    Returns the list of extracted file paths. Raises DownloadError for corrupt
    archives or unsafe member paths.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise DownloadError(f"Zip archive not found: {zip_path}")
    extract_dir = ensure_dir(extract_dir)
    root = extract_dir.resolve()
    extracted = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = (extract_dir / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise DownloadError(f"Unsafe path in archive: {member.filename}")
                if target.exists() and not overwrite:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted.append(target)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"Corrupt zip archive: {zip_path}") from exc
    return extracted


def verify_dataset(metadata_path, images_dir):
    """Check that every metadata image_id has a corresponding file on disk."""
    frame = load_metadata(metadata_path)
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise DownloadError(f"Images directory not found: {images_dir}")
    expected = {f"{image_id}.jpg" for image_id in frame["image_id"].astype(str)}
    on_disk = {path.name for path in images_dir.glob("*.jpg")}
    missing = sorted(expected - on_disk)
    extra = sorted(on_disk - expected)
    return {
        "n_metadata": len(frame),
        "n_on_disk": len(on_disk),
        "n_missing": len(missing),
        "missing_sample": missing[:10],
        "n_extra": len(extra),
    }


def download_ham10000(data_root="data", include_images=True, force=False, session=None, show_progress=True):
    """Download metadata and images, extract them, and verify the result."""
    data_root = Path(data_root)
    raw_dir = ensure_dir(data_root / "raw")
    images_dir = ensure_dir(data_root / "HAM10000" / "images")

    index = fetch_file_index(session)
    selection = select_files(index)
    source_metadata = raw_dir / selection["metadata"]
    metadata_entry = index[selection["metadata"]]
    try:
        download_file(
            DATAFILE_API.format(file_id=metadata_entry["id"]),
            source_metadata,
            metadata_entry["size"],
            metadata_entry["md5"],
            session=session,
            show_progress=show_progress,
        )
    except DownloadError as exc:
        raise DownloadError(f"Failed to download metadata file: {exc}") from exc

    metadata_path = raw_dir / "HAM10000_metadata.csv"
    if source_metadata.resolve() != metadata_path.resolve():
        normalize_metadata(source_metadata, metadata_path)

    results = {"metadata_csv": metadata_path, "images_dir": images_dir, "zips": {}}
    if include_images:
        for name in selection["images"]:
            entry = index[name]
            zip_path = raw_dir / name
            if force and zip_path.exists():
                zip_path.unlink()
            download_file(
                DATAFILE_API.format(file_id=entry["id"]),
                zip_path,
                entry["size"],
                entry["md5"],
                session=session,
                show_progress=show_progress,
            )
            extract_zip(zip_path, images_dir)
            results["zips"][name] = zip_path
        results["report"] = verify_dataset(metadata_path, images_dir)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download HAM10000 from Harvard Dataverse")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--metadata-only", action="store_true", help="Skip image archives")
    parser.add_argument("--force", action="store_true", help="Redownload archives from scratch")
    args = parser.parse_args(argv)

    try:
        results = download_ham10000(
            data_root=args.data_root, include_images=not args.metadata_only, force=args.force
        )
    except DownloadError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Metadata CSV: {results['metadata_csv']}")
    for name, path in results["zips"].items():
        print(f"Archive: {name} -> {path}")
    report = results.get("report")
    if report:
        print(
            f"Images on disk: {report['n_on_disk']} "
            f"(metadata rows: {report['n_metadata']}, missing: {report['n_missing']})"
        )
        if report["n_missing"] > 0:
            print(f"Missing sample: {report['missing_sample']}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
