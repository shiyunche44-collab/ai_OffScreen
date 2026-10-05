"""Naming characters and the human revision layer, over the real keyframes of the 30 s clip
(fake face analyzer and fake model)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.domain.index import Characters
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.services.app import AppServices
from offscreen.services.characters import CharacterEdit
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.pipeline import Pipeline, Providers
from offscreen.stages.analysis.characters import CENTROIDS_FILE, CHARACTERS_FILE

from .conftest import two_people_faces


@pytest.fixture
def providers(fakes: Providers) -> Providers:
    fakes.faces = two_people_faces()
    return fakes


@pytest.fixture
def services(cfg: AppConfig, providers: Providers, movie: Path) -> Iterator[AppServices]:
    with AppServices(
        cfg.model_copy(update={"media_roots": [movie.parent]}), providers=providers
    ) as s:
        yield s


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def build(services: AppServices, movie: Path, stage: str) -> str:
    asset = services.library.import_asset(str(movie))
    with Pipeline(services.cfg, services.jobs.providers, db=services.db) as p:
        p.run_stage(stage, asset.id)
    return asset.id


def llm_of(providers: Providers) -> FakeLLM:
    assert isinstance(providers.llm, FakeLLM)
    return providers.llm


def naming_calls(providers: Providers) -> list[Any]:
    return [c for c in llm_of(providers).calls if c[0] == "character_name"]


# --- the stage ---------------------------------------------------------------------------------


def test_naming_asks_once_with_faces_shots_and_dialogue_and_keeps_the_answer(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    asset = build(services, movie, "analysis.naming")
    ((_task, messages, version),) = naming_calls(providers)
    assert version == "character_name@1"
    (message,) = messages
    assert len(message.images) == 5  # three thumbnails of ch_01, two of ch_02
    for needle in ("人物 ch_01（出现在 4 个镜头）图 1–3", "人物 ch_02（出现在 2 个镜头）图 4–5",
                   "雪山里的画面", "We cannot stay here."):  # fmt: skip
        assert needle in message.content

    with Pipeline(services.cfg, providers, db=services.db) as p:
        art = p.peek("analysis.naming", asset)
        assert art is not None
        doc = art.read_model(CHARACTERS_FILE, Characters)
        assert art.meta == {"characters": 2, "named": 1, "asked": 2}
        first, second = doc.characters
        assert (first.name, first.name_source, first.aliases) == ("Sintel", "ai", ["the girl"])
        assert (first.role, first.bio) == ("主角", "寻找她的龙。")
        assert (second.name, second.name_source, second.role) == (None, None, "配角")
        # self-contained: the thumbnails and centres are in this artifact too
        assert first.face_cluster is not None
        for rel in (*first.face_cluster.thumbnails, CENTROIDS_FILE):
            assert art.path(rel).stat().st_size > 0


def test_a_name_without_a_real_quote_gets_one_repair_round_then_stays_unnamed(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    llm = llm_of(providers)
    real = llm._fns["character_name"]  # type: ignore[attr-defined]

    def liar(task: str, m: Any, schema: Any) -> Any:
        reply = real(task, m, schema)
        reply["characters"][0]["evidence"] = "Sintel is my name"  # not a line of the film
        return reply

    llm._fns["character_name"] = liar  # type: ignore[attr-defined]
    asset = build(services, movie, "analysis.naming")
    calls = naming_calls(providers)
    assert len(calls) == 2
    assert "不是台词里的原句" in calls[1][1][-1].content and "ch_01" in calls[1][1][-1].content

    with Pipeline(services.cfg, providers, db=services.db) as p:
        art = p.peek("analysis.naming", asset)
        assert art is not None
        first = art.read_model(CHARACTERS_FILE, Characters).characters[0]
        assert first.name is None and first.name_source is None
        assert first.role == "主角"  # the rest of the answer is kept
        assert art.meta["named"] == 0


def test_the_repair_round_can_fix_it(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    llm = llm_of(providers)
    real = llm._fns["character_name"]  # type: ignore[attr-defined]
    state = {"n": 0}

    def fixes_itself(task: str, m: Any, schema: Any) -> Any:
        reply = real(task, m, schema)
        state["n"] += 1
        if state["n"] == 1:
            reply["characters"][0]["evidence"] = "made up"
        return reply

    llm._fns["character_name"] = fixes_itself  # type: ignore[attr-defined]
    asset = build(services, movie, "analysis.naming")
    assert state["n"] == 2
    with Pipeline(services.cfg, providers, db=services.db) as p:
        art = p.peek("analysis.naming", asset)
        assert art is not None and art.meta["named"] == 1


def test_no_faces_means_no_request(
    cfg: AppConfig, fakes: Providers, movie: Path, services: AppServices
) -> None:
    from offscreen.providers.adapters.fake import FakeFaceAnalyzer

    fakes.faces = FakeFaceAnalyzer()
    asset = build(services, movie, "analysis.naming")
    assert naming_calls(fakes) == []
    with Pipeline(services.cfg, fakes, db=services.db) as p:
        art = p.peek("analysis.naming", asset)
        assert art is not None and art.meta == {"characters": 0, "named": 0, "asked": 0}


# --- reading and editing ----------------------------------------------------------------------


def test_view_before_anything_is_built_and_for_an_unknown_asset(
    services: AppServices, movie: Path
) -> None:
    asset = services.library.import_asset(str(movie))
    with pytest.raises(NotFound, match=r"analysis\.characters has not been built"):
        services.characters.view(asset.id)
    with pytest.raises(NotFound, match="unknown asset"):
        services.characters.view("ast_missing")


def test_view_shows_clusters_before_naming_and_names_after(
    services: AppServices, movie: Path
) -> None:
    asset = build(services, movie, "analysis.characters")
    view = services.characters.view(asset)
    assert view.named is False and [c.name for c in view.characters] == [None, None]
    thumb = view.characters[0].face_cluster.thumbnails[0]  # type: ignore[union-attr]
    assert thumb.startswith("artifacts/") and (services.cfg.data_dir / thumb).is_file()

    build(services, movie, "analysis.naming")
    view = services.characters.view(asset)
    assert view.named is True and [c.name for c in view.characters] == ["Sintel", None]


def test_human_edits_win_and_are_kept_apart_from_the_ai_output(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    asset = build(services, movie, "analysis.naming")
    view = services.characters.edit(
        asset, "ch_02", CharacterEdit(name="Dragon", aliases=["Scales"])
    )
    by = {c.id: c for c in view.characters}
    assert (by["ch_02"].name, by["ch_02"].name_source, by["ch_02"].aliases) == (
        "Dragon",
        "human",
        ["Scales"],
    )
    assert by["ch_01"].name == "Sintel" and by["ch_01"].name_source == "ai"

    path = services.cfg.data_dir / "overrides" / asset / "characters.overrides.json"
    assert path.is_file() and "Dragon" in path.read_text()
    with Pipeline(services.cfg, providers, db=services.db) as p:  # the AI document is untouched
        art = p.peek("analysis.naming", asset)
        assert art is not None
        assert art.read_model(CHARACTERS_FILE, Characters).characters[1].name is None

    again = services.characters.view(asset)  # and it persists
    assert again.characters[1].name == "Dragon"

    # editing a second field keeps the first
    view = services.characters.edit(asset, "ch_02", CharacterEdit(name="Dragon Jr"))
    assert view.characters[1].name == "Dragon Jr" and view.characters[1].aliases == ["Scales"]


def test_ignore_merge_and_reset(services: AppServices, movie: Path) -> None:
    asset = build(services, movie, "analysis.naming")
    view = services.characters.edit(asset, "ch_02", CharacterEdit(merged_into="ch_01"))
    assert [c.id for c in view.characters] == ["ch_01"] and view.merged == {"ch_02": "ch_01"}
    assert view.characters[0].face_cluster.size == 18  # type: ignore[union-attr]  # 12 + 6

    view = services.characters.edit(asset, "ch_02", CharacterEdit(reset=True))
    assert [c.id for c in view.characters] == ["ch_01", "ch_02"] and view.merged == {}

    view = services.characters.edit(asset, "ch_01", CharacterEdit(ignored=True))
    assert [c.id for c in view.characters] == ["ch_02"] and view.ignored == ["ch_01"]
    view = services.characters.edit(asset, "ch_01", CharacterEdit(ignored=False))
    assert [c.id for c in view.characters] == ["ch_01", "ch_02"]


def test_bad_edits_are_refused(services: AppServices, movie: Path) -> None:
    asset = build(services, movie, "analysis.characters")
    with pytest.raises(NotFound, match="unknown character ch_09"):
        services.characters.edit(asset, "ch_09", CharacterEdit(name="X"))
    with pytest.raises(InvalidInput, match="cannot merge ch_01 into ch_01"):
        services.characters.edit(asset, "ch_01", CharacterEdit(merged_into="ch_01"))
    with pytest.raises(InvalidInput, match="cannot merge ch_01 into ch_09"):
        services.characters.edit(asset, "ch_01", CharacterEdit(merged_into="ch_09"))


def test_edits_survive_a_re_clustering_that_renumbers_the_people(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    import numpy as np

    asset = build(services, movie, "analysis.characters")
    services.characters.edit(asset, "ch_02", CharacterEdit(name="Dragon"))

    # the faces are seen again in a different order: B is now found first (more often)
    def b_first(path: Path) -> list[Any]:
        from offscreen.providers.ports import DetectedFace

        shot = int(path.name.split("_")[1]) - 1
        rng = np.random.default_rng(3)
        a, b, _ = (v / np.linalg.norm(v) for v in rng.normal(size=(3, 32)))
        found = [DetectedFace((0.5, 0.1, 0.6, 0.3), 0.9, tuple(float(x) for x in b))]
        if shot >= 2:
            found += [DetectedFace((0.1, 0.1, 0.4, 0.5), 0.9, tuple(float(x) for x in a))]
        return found

    from offscreen.providers.adapters.fake import FakeFaceAnalyzer

    providers.faces = FakeFaceAnalyzer(b_first)  # B in all 12 keyframes, A in the last two shots
    providers.faces.id = "fake-faces@2"  # type: ignore[misc]  # another model: nothing is reused
    with Pipeline(services.cfg, providers, db=services.db) as p:
        p.run_stage("analysis.characters", asset)

    view = services.characters.view(asset)
    names = {c.id: c.name for c in view.characters}
    assert len(names) == 2 and view.unmatched_edits == 0
    # "Dragon" was the smaller, less frequent person (B in the first clustering); B is ch_01 now
    assert names == {"ch_01": "Dragon", "ch_02": None}


def test_an_edit_for_a_person_who_is_gone_is_kept_aside(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    from offscreen.providers.adapters.fake import FakeFaceAnalyzer
    from offscreen.providers.ports import DetectedFace

    asset = build(services, movie, "analysis.characters")
    services.characters.edit(asset, "ch_02", CharacterEdit(name="Dragon"))

    import numpy as np

    rng = np.random.default_rng(99)  # a film with one, entirely different, person
    z = rng.normal(size=32)
    providers.faces = FakeFaceAnalyzer(
        lambda p: [DetectedFace((0.1, 0.1, 0.4, 0.5), 0.9, tuple(float(x) for x in z))]
    )
    providers.faces.id = "fake-faces@2"  # type: ignore[misc]
    with Pipeline(services.cfg, providers, db=services.db) as p:
        p.run_stage("analysis.characters", asset)
    view = services.characters.view(asset)
    assert [c.name for c in view.characters] == [None] and view.unmatched_edits == 1

    # the orphan is still on disk, and comes back if the person does
    path = services.cfg.data_dir / "overrides" / asset / "characters.overrides.json"
    services.characters.edit(asset, "ch_01", CharacterEdit(name="Someone"))
    assert "Dragon" in path.read_text() and "Someone" in path.read_text()


# --- HTTP ---------------------------------------------------------------------------------------


def test_the_characters_route(client: TestClient, services: AppServices, movie: Path) -> None:
    asset = services.library.import_asset(str(movie)).id
    r = client.get(f"/api/assets/{asset}/index/characters")
    assert r.status_code == 404 and "has not been built" in r.json()["error"]["message"]

    build(services, movie, "analysis.naming")
    body = client.get(f"/api/assets/{asset}/index/characters").json()
    assert body["named"] is True and body["merged"] == {} and body["ignored"] == []
    assert [c["name"] for c in body["characters"]] == ["Sintel", None]
    thumb = body["characters"][0]["face_cluster"]["thumbnails"][0]
    assert client.get(f"/api/files/{thumb}").status_code == 200


def test_a_button_builds_the_characters_and_the_panel_edits_them(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    from offscreen.worker import Worker, WorkerSettings

    asset = services.library.import_asset(str(movie)).id
    r = client.post(f"/api/assets/{asset}/characters:build")
    assert r.status_code == 202
    job = r.json()
    assert (job["status"], job["stage"], job["lane"]) == ("queued", "analysis.naming", "cpu")
    assert (
        client.post(f"/api/assets/{asset}/characters:build").json()["id"] == job["id"]
    )  # no duplicate

    Worker(
        services.jobs.jobs,
        services.jobs.execute,
        services.cfg.data_dir,
        WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=0.0),
    ).drain(timeout_s=120)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "succeeded"

    body = client.get(f"/api/assets/{asset}/index/characters").json()
    assert [c["name"] for c in body["characters"]] == ["Sintel", None]

    r = client.patch(f"/api/assets/{asset}/characters/ch_02", json={"name": "Dragon"})
    assert r.status_code == 200
    assert [c["name"] for c in r.json()["characters"]] == ["Sintel", "Dragon"]
    assert r.json()["characters"][1]["name_source"] == "human"

    r = client.patch(f"/api/assets/{asset}/characters/ch_02", json={"merged_into": "ch_01"})
    assert [c["id"] for c in r.json()["characters"]] == ["ch_01"]
    assert r.json()["merged"] == {"ch_02": "ch_01"}
    r = client.patch(f"/api/assets/{asset}/characters/ch_02", json={"reset": True})
    assert [c["id"] for c in r.json()["characters"]] == ["ch_01", "ch_02"]

    bad = client.patch(f"/api/assets/{asset}/characters/ch_09", json={"name": "X"})
    assert bad.status_code == 404 and bad.json()["error"]["code"] == "not_found"
    bad = client.patch(f"/api/assets/{asset}/characters/ch_01", json={"merged_into": "ch_01"})
    assert bad.status_code == 422, bad.text
    assert (
        client.patch(f"/api/assets/{asset}/characters/ch_01", json={"ignored": "maybe"}).status_code
        == 422
    )
    assert client.post("/api/assets/ast_missing/characters:build").status_code == 404
