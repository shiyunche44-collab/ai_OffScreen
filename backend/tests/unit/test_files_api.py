"""/api/files/{path}: HTTP Range for players, and nothing outside the data directory."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.errors import NotFound
from offscreen.services.files import FileService, check_range

BODY = bytes(range(256)) * 4  # 1024 bytes, every byte value distinct per position mod 256


@pytest.fixture
def data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    art = d / "artifacts" / "output.render" / "abc123"
    art.mkdir(parents=True)
    (art / "final.mp4").write_bytes(BODY)
    (d / "logs").mkdir()
    (d / "logs" / "job_1.log").write_text("hello\n", encoding="utf-8")
    (d / "notes").mkdir()
    (d / "notes" / "a b.txt").write_text("spaced name", encoding="utf-8")
    # things that must stay out of reach
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    evil = tmp_path / "data_evil"  # shares the "data" prefix
    evil.mkdir()
    (evil / "secret.txt").write_text("sibling", encoding="utf-8")
    return d


@pytest.fixture
def client(data: Path) -> Iterator[TestClient]:
    cfg = config(data)
    with AppServices(cfg) as s, TestClient(create_app(s), raise_server_exceptions=False) as c:
        yield c


def config(data_dir: Path) -> AppConfig:
    return AppConfig.model_validate({"data_dir": str(data_dir), "tts": {"provider": "edge_tts"}})


VIDEO = "/api/files/artifacts/output.render/abc123/final.mp4"


# --- serving ----------------------------------------------------------------------------


def test_serves_a_file_with_type_and_range_support(client: TestClient) -> None:
    r = client.get(VIDEO)
    assert r.status_code == 200 and r.content == BODY
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-length"] == str(len(BODY))


def test_artifacts_are_cacheable_forever_other_files_are_not(client: TestClient) -> None:
    assert "immutable" in client.get(VIDEO).headers["cache-control"]
    assert "cache-control" not in client.get("/api/files/logs/job_1.log").headers


def test_head_returns_headers_without_a_body(client: TestClient) -> None:
    r = client.head(VIDEO)
    assert r.status_code == 200 and r.content == b""
    assert r.headers["content-length"] == str(len(BODY))


def test_names_with_spaces_work(client: TestClient) -> None:
    assert client.get("/api/files/notes/a%20b.txt").text == "spaced name"


@pytest.mark.parametrize(
    ("header", "start", "end"),
    [
        ("bytes=0-9", 0, 9),
        ("bytes=100-199", 100, 199),
        ("bytes=1000-", 1000, 1023),  # open ended
        ("bytes=-24", 1000, 1023),  # suffix
        ("bytes=1020-5000", 1020, 1023),  # end past EOF is clamped
        ("bytes=0-0", 0, 0),
    ],
)
def test_range_requests_return_206_with_exactly_those_bytes(
    client: TestClient, header: str, start: int, end: int
) -> None:
    r = client.get(VIDEO, headers={"Range": header})
    assert r.status_code == 206
    assert r.content == BODY[start : end + 1]
    assert r.headers["content-range"] == f"bytes {start}-{end}/{len(BODY)}"
    assert r.headers["content-length"] == str(end - start + 1)


def test_unsatisfiable_range_is_a_416_in_the_error_shape(client: TestClient) -> None:
    r = client.get(VIDEO, headers={"Range": "bytes=5000-6000"})
    assert r.status_code == 416
    assert r.json()["error"]["code"] == "range_not_satisfiable"
    assert r.headers["content-range"] == f"bytes */{len(BODY)}"


def test_malformed_range_is_a_400_in_the_error_shape(client: TestClient) -> None:
    for bad in ("bytes=abc", "pages=1-2", "bytes=", "bytes=-", "bytes=9-3", "garbage"):
        r = client.get(VIDEO, headers={"Range": bad})
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request", bad


def test_multiple_ranges_are_served_as_multipart(client: TestClient) -> None:
    r = client.get(VIDEO, headers={"Range": "bytes=0-1,10-11"})
    assert r.status_code == 206
    assert r.headers["content-type"].startswith("multipart/byteranges")


@pytest.mark.parametrize(
    ("header", "size", "verdict"),
    [
        ("bytes=0-0", 1, "ok"),
        ("bytes=0-", 0, "unsatisfiable"),
        ("bytes=-5", 0, "unsatisfiable"),
        ("bytes=-0", 10, "unsatisfiable"),
        ("bytes=10-", 10, "unsatisfiable"),
        ("bytes=9-", 10, "ok"),
        ("bytes=20-30,0-1", 10, "ok"),  # one satisfiable range is enough
        ("bytes=20-30,40-", 10, "unsatisfiable"),
        ("BYTES=0-1", 10, "ok"),
        ("bytes=1-2-3", 10, "malformed"),
        ("bytes=0-1,", 10, "malformed"),
    ],
)
def test_check_range(header: str, size: int, verdict: str) -> None:
    assert check_range(header, size) == verdict


def test_missing_file_and_directory_are_404(client: TestClient) -> None:
    for path in (
        "/api/files/nope.mp4",
        "/api/files/artifacts",
        "/api/files/artifacts/output.render/abc123",
    ):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", path


# --- confinement ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/files/../secret.txt",
        "/api/files/%2e%2e/secret.txt",
        "/api/files/%2E%2E/%2E%2E/etc/passwd",
        "/api/files/logs/../../secret.txt",
        "/api/files/logs/%2e%2e/%2e%2e/secret.txt",
        "/api/files/..%2fsecret.txt",
        "/api/files/%2e%2e%2fsecret.txt",
        "/api/files/../data_evil/secret.txt",
        "/api/files/%2e%2e/data_evil/secret.txt",
        "/api/files//etc/passwd",  # absolute path smuggled through the double slash
        "/api/files/%2fetc/passwd",
        "/api/files/logs%5c..%5c..%5csecret.txt",  # backslashes
        "/api/files/logs/job_1.log%00.png",  # NUL byte
        "/api/files/%00",
    ],
)
def test_nothing_outside_the_data_directory_is_reachable(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 404, (path, r.status_code)
    assert b"outside" not in r.content and b"sibling" not in r.content and b"root:" not in r.content


def test_a_symlink_pointing_out_of_the_data_directory_is_refused(
    client: TestClient, data: Path, tmp_path: Path
) -> None:
    os.symlink(tmp_path / "secret.txt", data / "link.txt")
    os.symlink(tmp_path / "data_evil", data / "linkdir")
    assert client.get("/api/files/link.txt").status_code == 404
    assert client.get("/api/files/linkdir/secret.txt").status_code == 404


def test_a_symlink_staying_inside_is_fine(client: TestClient, data: Path) -> None:
    os.symlink(data / "logs" / "job_1.log", data / "alias.log")
    assert client.get("/api/files/alias.log").text == "hello\n"


def test_the_database_is_never_served(client: TestClient, data: Path) -> None:
    for name in ("offscreen.db", "offscreen.db-wal", "offscreen.db-shm", "x.SQLITE"):
        (data / name).write_bytes(b"SQLite format 3\0")
        assert client.get(f"/api/files/{name}").status_code == 404, name


def test_resolve_unit_cases(data: Path) -> None:
    svc = FileService(config(data))
    assert svc.resolve("logs/job_1.log") == (data / "logs" / "job_1.log").resolve()
    for bad in ("", ".", "..", "../secret.txt", "/etc/passwd", "logs", "a\0b", "a\\b", "x/../../y"):
        with pytest.raises(NotFound):
            svc.resolve(bad)


def test_a_relative_data_dir_works_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("ok")
    (tmp_path / "secret.txt").write_text("outside")
    svc = FileService(config(Path("./data")))
    assert svc.resolve("x.txt").read_text() == "ok"
    with pytest.raises(NotFound):
        svc.resolve("../secret.txt")
