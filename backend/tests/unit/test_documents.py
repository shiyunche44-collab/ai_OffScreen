"""Versioned documents: the append-only store with its optimistic lock, the segment diff, and
the script endpoints on top (ARCHITECTURE §6.5)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from offscreen.algo.docdiff import diff_documents
from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.domain.project import ProjectOptions
from offscreen.domain.script import Script, ScriptContent
from offscreen.providers.adapters.fake import FakeFaceAnalyzer, FakeLLM, FakeTTS
from offscreen.services.app import AppServices
from offscreen.services.errors import Conflict, InvalidInput, NotFound
from offscreen.services.pipeline import Providers
from offscreen.store.documents import StaleBase
from offscreen.store.repos import AssetRepo

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "domain"


@pytest.fixture
def services(tmp_path: Path) -> Iterator[AppServices]:
    cfg = AppConfig.model_validate(
        {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
    )
    fakes = Providers(llm=FakeLLM({}), tts=FakeTTS(), detector=None, faces=FakeFaceAnalyzer())  # type: ignore[arg-type]
    with AppServices(cfg, providers=fakes) as s:
        yield s


@pytest.fixture
def project_id(services: AppServices) -> str:
    asset = MediaAsset.model_validate(json.loads((FIXTURES / "asset.json").read_text("utf-8")))
    AssetRepo(services.db).add(asset)
    return services.library.create_project(asset.id, "p", ProjectOptions()).id


@pytest.fixture
def content() -> ScriptContent:
    script = Script.model_validate(json.loads((FIXTURES / "script.json").read_text("utf-8")))
    return ScriptContent.of(script)


def edited(content: ScriptContent, text: str) -> ScriptContent:
    first = content.segments[0].model_copy(update={"text": text})
    return content.model_copy(update={"segments": [first, *content.segments[1:]]})


# --- service: history and the optimistic lock ------------------------------------------


def test_first_save_creates_version_one_and_later_ones_chain(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    docs = services.documents
    assert docs.script_head(project_id) is None
    with pytest.raises(NotFound):
        docs.script(project_id)

    v1 = docs.save_script(project_id, content, None, author="ai")
    v2 = docs.save_script(project_id, edited(content, "新的开场白。"), 1)
    assert (v1.version, v1.parent_version, v1.author) == (1, None, "ai")
    assert (v2.version, v2.parent_version, v2.author) == (2, 1, "human")
    assert v2.id == v1.id and v2.id.startswith("scr_")  # one document, many versions
    assert docs.script(project_id).segments[0].text == "新的开场白。"
    assert docs.script(project_id, 1).segments[0].text == content.segments[0].text
    assert [(v.version, v.parent_version, v.author) for v in docs.script_versions(project_id)] == [
        (2, 1, "human"),
        (1, None, "ai"),
    ]


def test_versions_are_immutable_files(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    docs = services.documents
    docs.save_script(project_id, content, None)
    path = services.cfg.data_dir / "projects" / project_id / "docs" / "script" / "v1.json"
    before = path.read_bytes()
    docs.save_script(project_id, edited(content, "改过了。"), 1)
    assert path.read_bytes() == before
    assert (path.parent / "v2.json").exists()


def test_a_stale_base_version_is_a_conflict_and_changes_nothing(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    docs = services.documents
    docs.save_script(project_id, content, None)
    docs.save_script(project_id, edited(content, "甲"), 1)

    with pytest.raises(Conflict, match="current 2"):
        docs.save_script(project_id, edited(content, "乙"), 1)
    with pytest.raises(Conflict):  # claiming there is no script yet
        docs.save_script(project_id, content, None)
    assert docs.script_head(project_id) == 2
    assert not (services.cfg.data_dir / "projects" / project_id / "docs/script/v3.json").exists()


def test_an_invalid_script_is_rejected_without_a_version(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    dup = content.model_copy(update={"segments": [content.segments[0], content.segments[0]]})
    with pytest.raises(InvalidInput, match="duplicate"):
        services.documents.save_script(project_id, dup, None)
    assert services.documents.script_head(project_id) is None
    services.documents.save_script(project_id, content, None)  # base None is still right


def test_restore_makes_a_new_version_with_the_old_content(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    docs = services.documents
    docs.save_script(project_id, content, None)
    docs.save_script(project_id, edited(content, "改动"), 1)
    restored = docs.restore_script(project_id, 1, 2)
    assert (restored.version, restored.parent_version) == (3, 2)
    assert ScriptContent.of(restored) == content
    with pytest.raises(NotFound):
        docs.restore_script(project_id, 9, 3)


def test_unknown_project_is_not_found(services: AppServices, content: ScriptContent) -> None:
    with pytest.raises(NotFound):
        services.documents.save_script("prj_missing", content, None)
    with pytest.raises(NotFound):
        services.documents.script_versions("prj_missing")


def test_racing_writers_cannot_both_win(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    services.documents.save_script(project_id, content, None)
    outcomes: list[str] = []
    lock = threading.Lock()
    gate = threading.Barrier(6)

    def writer(n: int) -> None:
        gate.wait()
        try:
            services.documents.save_script(project_id, edited(content, f"写手{n}"), 1)
            result = "won"
        except Conflict:
            result = "lost"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(outcomes) == ["lost"] * 5 + ["won"]
    assert [v.version for v in services.documents.script_versions(project_id)] == [2, 1]


def test_store_reports_the_actual_head_on_a_stale_base(
    services: AppServices, project_id: str, content: ScriptContent
) -> None:
    services.documents.save_script(project_id, content, None)
    with pytest.raises(StaleBase) as err:
        services.documents.store.append(
            project_id, "script", 5, lambda *_: pytest.fail("not built"), author="human"
        )
    assert (err.value.base_version, err.value.current) == (5, 1)


# --- diff -------------------------------------------------------------------------------


def seg(id_: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"id": id_, "kind": "narration", "text": text, "scene_refs": ["sc_1"], **extra}


def doc(*segments: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {"id": "scr_1", "version": 1, "author": "ai", "params": {"style": "x"}, **top} | {
        "segments": list(segments)
    }


def test_diff_classifies_segments_by_id() -> None:
    a = doc(seg("seg_1", "甲"), seg("seg_2", "乙"), seg("seg_3", "丙"))
    b = doc(seg("seg_1", "甲"), seg("seg_2", "乙乙", scene_refs=["sc_2"]), seg("seg_4", "丁"))
    d = diff_documents("script", 1, 2, a, b)
    assert [(c.segment_id, c.status, c.fields) for c in d.changes] == [
        ("seg_1", "unchanged", []),
        ("seg_2", "changed", ["scene_refs", "text"]),
        ("seg_4", "added", []),
        ("seg_3", "removed", []),
    ]
    assert not d.reordered and d.params_changed == []


def test_diff_notices_reordering_and_param_changes_but_not_version_metadata() -> None:
    a = doc(seg("seg_1", "甲"), seg("seg_2", "乙"))
    b = doc(
        seg("seg_2", "乙"), seg("seg_1", "甲"), version=2, author="human", params={"style": "y"}
    )
    d = diff_documents("script", 1, 2, a, b)
    assert d.reordered
    assert d.params_changed == ["params"]
    assert {c.status for c in d.changes} == {"unchanged"}


def test_adding_a_segment_is_not_a_reorder() -> None:
    a = doc(seg("seg_1", "甲"), seg("seg_2", "乙"))
    b = doc(seg("seg_1", "甲"), seg("seg_9", "新"), seg("seg_2", "乙"))
    assert not diff_documents("script", 1, 2, a, b).reordered


# --- API --------------------------------------------------------------------------------


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def body(content: ScriptContent, base: int | None) -> dict[str, Any]:
    return {**content.model_dump(mode="json"), "base_version": base}


def test_api_save_read_history_diff_restore(
    client: TestClient, project_id: str, content: ScriptContent
) -> None:
    url = f"/api/projects/{project_id}/script"
    assert client.get(url).status_code == 404  # nothing generated, nothing saved

    first = client.put(url, json=body(content, None))
    assert first.status_code == 200 and first.json()["version"] == 1
    second = client.put(url, json=body(edited(content, "第二版"), 1))
    assert second.json()["version"] == 2 and second.json()["author"] == "human"

    assert client.get(url).json()["version"] == 2
    assert client.get(url, params={"version": 1}).json()["segments"][0]["text"] != "第二版"
    assert client.get(url, params={"version": 7}).status_code == 404
    history = client.get(f"{url}/versions").json()
    assert [(v["version"], v["parent_version"]) for v in history] == [(2, 1), (1, None)]

    diff = client.get(f"{url}/diff", params={"a": 1, "b": 2}).json()
    assert diff["changes"][0] == {"segment_id": "seg_01", "status": "changed", "fields": ["text"]}

    restored = client.post(f"{url}:restore", json={"version": 1, "base_version": 2})
    assert restored.status_code == 200 and restored.json()["version"] == 3


def test_api_stale_save_is_409_with_the_error_shape(
    client: TestClient, project_id: str, content: ScriptContent
) -> None:
    url = f"/api/projects/{project_id}/script"
    client.put(url, json=body(content, None))
    client.put(url, json=body(edited(content, "A"), 1))
    r = client.put(url, json=body(edited(content, "B"), 1))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"
    assert client.get(url).json()["segments"][0]["text"] == "A"


def test_api_invalid_script_is_422(
    client: TestClient, project_id: str, content: ScriptContent
) -> None:
    bad = body(content, None)
    bad["segments"][0]["scene_refs"] = []
    assert client.put(f"/api/projects/{project_id}/script", json=bad).status_code == 422
