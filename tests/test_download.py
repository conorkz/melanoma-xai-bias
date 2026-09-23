"""Tests for melanoma.data.download with fully mocked network sessions."""

import hashlib
import shutil
import zipfile
from pathlib import Path

import cv2
import pytest
import requests

from melanoma.data.download import (
    DownloadError,
    download_file,
    extract_zip,
    fetch_file_index,
    md5sum,
    normalize_metadata,
    select_files,
    verify_dataset,
)
from melanoma.data.metadata import load_metadata

from conftest import make_image, make_synthetic_dataset

EMPTY_MD5 = hashlib.md5(b"").hexdigest()
HELLO_MD5 = hashlib.md5(b"hello").hexdigest()


class FakeResponse:
    def __init__(self, status_code=200, content=b"", json_data=None, headers=None):
        self.status_code = status_code
        self._content = content
        self._json = json_data
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json payload")
        return self._json

    def iter_content(self, chunk_size=1024):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responder(url, kwargs)


def file_entry(name, file_id, size, md5=None, checksum_only=False):
    payload = {"filename": name, "id": file_id, "filesize": size}
    if md5 is not None:
        if checksum_only:
            payload["checksum"] = {"type": "MD5", "value": md5}
        else:
            payload["md5"] = md5
    return {"dataFile": payload}


def api_payload(entries):
    return {"data": {"latestVersion": {"files": entries}}}


def test_fetch_file_index_sends_user_agent():
    entries = [file_entry("HAM10000_metadata.csv", 1, 100)]
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data=api_payload(entries)))
    fetch_file_index(session=session)
    assert "User-Agent" in session.calls[0][1]["headers"]


def test_download_file_sends_user_agent(tmp_path):
    session = FakeSession(lambda url, kwargs: FakeResponse(content=b"abc"))
    download_file(
        "http://example.com/f", tmp_path / "f.bin", session=session, show_progress=False
    )
    assert "User-Agent" in session.calls[0][1]["headers"]


def test_fetch_file_index_parses_entries():
    entries = [
        file_entry("HAM10000_metadata.csv", 1, 100, HELLO_MD5),
        file_entry("HAM10000_images_part_1.zip", 2, 200),
    ]
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data=api_payload(entries)))
    index = fetch_file_index(session=session)
    assert index["HAM10000_metadata.csv"]["id"] == 1
    assert index["HAM10000_metadata.csv"]["md5"] == HELLO_MD5
    assert index["HAM10000_images_part_1.zip"]["md5"] is None


def test_fetch_file_index_checksum_fallback():
    entries = [file_entry("HAM10000_metadata.tab", 1, 100, HELLO_MD5, checksum_only=True)]
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data=api_payload(entries)))
    index = fetch_file_index(session=session)
    assert index["HAM10000_metadata.tab"]["md5"] == HELLO_MD5


def test_fetch_file_index_uses_label_when_filename_missing():
    entries = [{"label": "from_label.csv", "dataFile": {"id": 5, "filesize": 10}}]
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data=api_payload(entries)))
    index = fetch_file_index(session=session)
    assert "from_label.csv" in index


def test_fetch_file_index_skips_incomplete_entries():
    entries = [file_entry("ok.zip", 1, 10), {"dataFile": {"name": "no_id.zip"}}]
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data=api_payload(entries)))
    index = fetch_file_index(session=session)
    assert list(index) == ["ok.zip"]


def test_fetch_file_index_http_error():
    session = FakeSession(lambda url, kwargs: FakeResponse(status_code=500))
    with pytest.raises(DownloadError, match="500"):
        fetch_file_index(session=session)


def test_fetch_file_index_invalid_json():
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data=None))
    with pytest.raises(DownloadError, match="JSON"):
        fetch_file_index(session=session)


def test_fetch_file_index_no_files():
    session = FakeSession(lambda url, kwargs: FakeResponse(json_data={"data": {}}))
    with pytest.raises(DownloadError):
        fetch_file_index(session=session)


def test_fetch_file_index_connection_error():
    def responder(url, kwargs):
        raise requests.ConnectionError("boom")

    with pytest.raises(DownloadError, match="failed"):
        fetch_file_index(session=FakeSession(responder))


def test_select_files_finds_required_files():
    index = {
        "HAM10000_metadata.csv": {"id": 1, "size": 10, "md5": None},
        "HAM10000_images_part_1.zip": {"id": 2, "size": 10, "md5": None},
        "HAM10000_images_part_2.zip": {"id": 3, "size": 10, "md5": None},
    }
    selection = select_files(index)
    assert selection["metadata"] == "HAM10000_metadata.csv"
    assert selection["images"] == ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]


def test_select_files_accepts_tab_metadata_and_ignores_other_zips():
    index = {
        "HAM10000_metadata.tab": {"id": 1, "size": 10, "md5": None},
        "HAM10000_images_part_1.zip": {"id": 2, "size": 10, "md5": None},
        "HAM10000_images_part_2.zip": {"id": 3, "size": 10, "md5": None},
        "HAM10000_segmentations_lesion_tschandl.zip": {"id": 4, "size": 10, "md5": None},
    }
    selection = select_files(index)
    assert selection["metadata"] == "HAM10000_metadata.tab"
    assert len(selection["images"]) == 2
    assert all("images_part" in name for name in selection["images"])


def test_select_files_missing_csv_raises():
    with pytest.raises(DownloadError):
        select_files({"a.zip": {"id": 1}})


def test_select_files_single_zip_raises():
    index = {
        "HAM10000_metadata.csv": {"id": 1},
        "HAM10000_images_part_1.zip": {"id": 2},
    }
    with pytest.raises(DownloadError, match="two"):
        select_files(index)


def test_md5sum():
    path = Path(__file__)
    assert md5sum(path) == hashlib.md5(path.read_bytes()).hexdigest()


def test_download_file_cached_skips_network(tmp_path):
    content = b"hello"
    dest = tmp_path / "file.bin"
    dest.write_bytes(content)

    def responder(url, kwargs):
        raise AssertionError("network must not be used for cached files")

    session = FakeSession(responder)
    result = download_file(
        "http://example.com/file",
        dest,
        expected_size=len(content),
        expected_md5=HELLO_MD5,
        session=session,
        show_progress=False,
    )
    assert result == "cached"
    assert session.calls == []


def test_download_file_success(tmp_path):
    content = b"abc" * 100
    dest = tmp_path / "out" / "file.bin"
    session = FakeSession(
        lambda url, kwargs: FakeResponse(content=content, headers={"Content-Length": len(content)})
    )
    result = download_file(
        "http://example.com/file",
        dest,
        expected_size=len(content),
        expected_md5=hashlib.md5(content).hexdigest(),
        session=session,
        show_progress=False,
    )
    assert result == "downloaded"
    assert dest.read_bytes() == content
    assert not (tmp_path / "out" / "file.bin.part").exists()


def test_download_file_http_error(tmp_path):
    session = FakeSession(lambda url, kwargs: FakeResponse(status_code=404))
    with pytest.raises(DownloadError, match="404"):
        download_file("http://example.com/missing", tmp_path / "f.bin", session=session, show_progress=False)


def test_download_file_short_download_raises_and_keeps_part(tmp_path):
    dest = tmp_path / "f.bin"
    session = FakeSession(lambda url, kwargs: FakeResponse(content=b"abc"))
    with pytest.raises(DownloadError, match="Incomplete"):
        download_file(
            "http://example.com/f", dest, expected_size=999, session=session, show_progress=False
        )
    assert not dest.exists()
    assert dest.with_suffix(".bin.part").exists()


def test_download_file_md5_mismatch_removes_part(tmp_path):
    dest = tmp_path / "f.bin"
    session = FakeSession(lambda url, kwargs: FakeResponse(content=b"abc"))
    with pytest.raises(DownloadError, match="MD5"):
        download_file(
            "http://example.com/f",
            dest,
            expected_size=3,
            expected_md5=EMPTY_MD5,
            session=session,
            show_progress=False,
        )
    assert not dest.exists()
    assert not dest.with_suffix(".bin.part").exists()


def test_download_file_resumes_partial(tmp_path):
    full = b"0123456789abcdef"
    dest = tmp_path / "f.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(full[:5])

    def responder(url, kwargs):
        assert kwargs["headers"].get("Range") == "bytes=5-"
        return FakeResponse(status_code=206, content=full[5:])

    session = FakeSession(responder)
    result = download_file(
        "http://example.com/f",
        dest,
        expected_size=len(full),
        session=session,
        show_progress=False,
    )
    assert result == "downloaded"
    assert dest.read_bytes() == full
    assert not part.exists()


def test_download_file_restarts_when_server_ignores_range(tmp_path):
    full = b"0123456789abcdef"
    dest = tmp_path / "f.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(full[:5])

    def responder(url, kwargs):
        assert "Range" in kwargs["headers"]
        return FakeResponse(status_code=200, content=full)

    session = FakeSession(responder)
    result = download_file(
        "http://example.com/f",
        dest,
        expected_size=len(full),
        session=session,
        show_progress=False,
    )
    assert result == "downloaded"
    assert dest.read_bytes() == full


def test_download_file_md5_takes_precedence_over_size(tmp_path):
    content = b"abc"
    dest = tmp_path / "f.bin"
    session = FakeSession(lambda url, kwargs: FakeResponse(content=content))
    result = download_file(
        "http://example.com/f",
        dest,
        expected_size=9999,
        expected_md5=hashlib.md5(content).hexdigest(),
        session=session,
        show_progress=False,
    )
    assert result == "downloaded"
    assert dest.read_bytes() == content


def test_download_file_restarts_on_range_not_satisfiable(tmp_path):
    full = b"0123456789abcdef"
    dest = tmp_path / "f.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(full[:5])

    def responder(url, kwargs):
        if "Range" in kwargs["headers"]:
            return FakeResponse(status_code=416, content=b"")
        return FakeResponse(status_code=200, content=full)

    session = FakeSession(responder)
    result = download_file(
        "http://example.com/f",
        dest,
        expected_size=len(full),
        session=session,
        show_progress=False,
    )
    assert result == "downloaded"
    assert dest.read_bytes() == full
    assert not part.exists()


def test_download_file_connection_error(tmp_path):
    def responder(url, kwargs):
        raise requests.ConnectionError("down")

    with pytest.raises(DownloadError):
        download_file("http://example.com/f", tmp_path / "f.bin", session=FakeSession(responder), show_progress=False)


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_extract_zip_success(tmp_path):
    archive = make_zip(tmp_path / "a.zip", {"x.txt": "one", "nested/y.txt": "two"})
    extracted = extract_zip(archive, tmp_path / "out")
    assert len(extracted) == 2
    assert (tmp_path / "out" / "x.txt").read_text() == "one"
    assert (tmp_path / "out" / "nested" / "y.txt").read_text() == "two"


def test_extract_zip_skips_existing(tmp_path):
    archive = make_zip(tmp_path / "a.zip", {"x.txt": "new"})
    target = tmp_path / "out" / "x.txt"
    target.parent.mkdir(parents=True)
    target.write_text("old")
    extract_zip(archive, tmp_path / "out")
    assert target.read_text() == "old"
    extract_zip(archive, tmp_path / "out", overwrite=True)
    assert target.read_text() == "new"


def test_extract_zip_corrupt_raises(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"definitely not a zip")
    with pytest.raises(DownloadError, match="Corrupt"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_missing_raises(tmp_path):
    with pytest.raises(DownloadError, match="not found"):
        extract_zip(tmp_path / "missing.zip", tmp_path / "out")


def test_extract_zip_rejects_path_traversal(tmp_path):
    archive = make_zip(tmp_path / "evil.zip", {"../evil.txt": "gotcha"})
    with pytest.raises(DownloadError, match="Unsafe"):
        extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_verify_dataset_complete(synthetic_dataset):
    report = verify_dataset(synthetic_dataset["metadata_csv"], synthetic_dataset["images_dir"])
    assert report["n_missing"] == 0
    assert report["n_extra"] == 0
    assert report["n_on_disk"] == report["n_metadata"]


def test_verify_dataset_reports_missing_and_extra(synthetic_dataset):
    (synthetic_dataset["images_dir"] / "unexpected.jpg").write_bytes(b"x")
    first = synthetic_dataset["frame"].iloc[0]["image_id"]
    (synthetic_dataset["images_dir"] / f"{first}.jpg").unlink()
    report = verify_dataset(synthetic_dataset["metadata_csv"], synthetic_dataset["images_dir"])
    assert report["n_missing"] == 1
    assert first in report["missing_sample"][0]
    assert report["n_extra"] == 1


def test_verify_dataset_missing_images_dir(synthetic_dataset):
    with pytest.raises(DownloadError):
        verify_dataset(synthetic_dataset["metadata_csv"], synthetic_dataset["root"] / "nope")


def test_normalize_metadata_converts_tab(tmp_path):
    source = tmp_path / "meta.tab"
    source.write_text("image_id\tdx\tlesion_id\nimg_0000\tmel\tl1\n", encoding="utf-8")
    destination = tmp_path / "meta.csv"
    normalize_metadata(source, destination)
    frame = load_metadata(destination)
    assert list(frame["image_id"]) == ["img_0000"]
    assert frame.iloc[0]["dx"] == "mel"


def test_normalize_metadata_keeps_comma_files(tmp_path):
    source = tmp_path / "meta.csv"
    source.write_text("image_id,dx,lesion_id\nimg_0000,nv,l1\n", encoding="utf-8")
    destination = tmp_path / "out.csv"
    normalize_metadata(source, destination)
    frame = load_metadata(destination)
    assert frame.iloc[0]["dx"] == "nv"


def test_normalize_metadata_missing_source_raises(tmp_path):
    with pytest.raises(DownloadError):
        normalize_metadata(tmp_path / "missing.tab", tmp_path / "out.csv")


def test_download_ham10000_orchestration(tmp_path, monkeypatch):
    import melanoma.data.download as download_module

    index = {
        "HAM10000_metadata.tab": {"id": 1, "size": 10, "md5": None},
        "HAM10000_images_part_1.zip": {"id": 2, "size": 10, "md5": None},
        "HAM10000_images_part_2.zip": {"id": 3, "size": 10, "md5": None},
    }
    monkeypatch.setattr(
        download_module, "fetch_file_index", lambda session=None, timeout=60: index
    )

    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("img_0000.jpg", cv2.imencode(".jpg", make_image(1))[1].tobytes())

    def fake_download(url, dest, expected_size=None, expected_md5=None, session=None, **kwargs):
        assert "format=original" in url
        dest = Path(dest)
        if dest.name == "HAM10000_metadata.tab":
            dest.write_text("image_id\tdx\tlesion_id\nimg_0000\tmel\tl1\n", encoding="utf-8")
        else:
            shutil.copyfile(zip_path, dest)
        return "downloaded"

    monkeypatch.setattr(download_module, "download_file", fake_download)

    results = download_module.download_ham10000(data_root=tmp_path / "data")
    assert results["metadata_csv"] == tmp_path / "data" / "raw" / "HAM10000_metadata.csv"
    frame = load_metadata(results["metadata_csv"])
    assert list(frame["image_id"]) == ["img_0000"]
    report = results["report"]
    assert report["n_on_disk"] == 1
    assert report["n_missing"] == 0
    assert report["n_extra"] == 0
