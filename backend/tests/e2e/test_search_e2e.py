"""Searching the shots of the 30 s clip: embeddings stage, derived LanceDB index, service and
route. Fake embedder: a sentence and a picture are close when they share words, so what a search
should find is known."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.domain.index import ShotIndex
from offscreen.providers.adapters.fake import FakeEmbedder, FakeLLM
from offscreen.services.app import AppServices
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.pipeline import Pipeline, Providers
from offscreen.stages.analysis.embeddings import (
    IMAGE_VECTORS_FILE,
    SHOT_INDEX_FILE,
    TEXT_VECTORS_FILE,
    EmbeddingsError,
)


def picture_of(path: Path) -> str:
    """What the fake 'sees' in a keyframe: the words its shot is described with."""
    shot = re.search(r"sh_\d+", path.name)
    assert shot is not None
    return f"雪山里的画面 {shot.group(0)}"


@pytest.fixture
def providers(fakes: Providers) -> Providers:
    fakes.image_embedder = FakeEmbedder(dim=256, image_of=picture_of, id="fake-clip@1")
    fakes.text_embedder = FakeEmbedder(dim=192, id="fake-bge@1", images=False)
    llm = fakes.llm
    assert isinstance(llm, FakeLLM)
    real = llm._fns["shot_caption"]  # type: ignore[attr-defined]

    def with_credits(task: str, m: Any, schema: Any) -> Any:  # the last shot is the end titles
        reply = real(task, m, schema)
        for c in reply["captions"]:
            if c["shot_id"].endswith("4"):
                c["is_credits"] = True
        return reply

    llm._fns["shot_caption"] = with_credits  # type: ignore[attr-defined]
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


@pytest.fixture
def asset(services: AppServices, providers: Providers, movie: Path) -> str:
    a = services.library.import_asset(str(movie))
    with Pipeline(services.cfg, providers, db=services.db) as p:
        p.run_stage("analysis.embeddings", a.id)
    return a.id


# --- the stage ---------------------------------------------------------------------------------


def test_the_stage_writes_entries_and_unit_vectors_in_shot_order(
    services: AppServices, providers: Providers, asset: str
) -> None:
    with Pipeline(services.cfg, providers, db=services.db) as p:
        art = p.peek("analysis.embeddings", asset)
    assert art is not None and art.meta == {"shots": 4, "image_dim": 256, "text_dim": 192}
    doc = art.read_model(SHOT_INDEX_FILE, ShotIndex)
    assert (doc.image_model, doc.text_model) == ("fake-clip@1", "fake-bge@1")
    assert [e.shot_id[-1] for e in doc.shots] == ["1", "2", "3", "4"]
    first = doc.shots[0]
    assert first.scene_id is not None and first.caption.startswith("雪山里的画面")
    assert [e.is_credits for e in doc.shots] == [False, False, False, True]
    for name, dim in ((IMAGE_VECTORS_FILE, 256), (TEXT_VECTORS_FILE, 192)):
        m = np.load(art.path(name))
        assert m.shape == (4, dim) and m.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    image = providers.image_embedder
    assert isinstance(image, FakeEmbedder) and len(image.image_calls[0]) == 4
    assert all("_b.jpg" in p.name for p in image.image_calls[0])  # the middle keyframe


def test_either_embedder_alone_is_enough_and_none_is_an_error(
    services: AppServices, providers: Providers, movie: Path
) -> None:
    providers.image_embedder = None
    a = services.library.import_asset(str(movie)).id
    with Pipeline(services.cfg, providers, db=services.db) as p:
        art = p.run_stage("analysis.embeddings", a).artifact
        assert not (art.dir / IMAGE_VECTORS_FILE).exists()
        assert (art.dir / TEXT_VECTORS_FILE).exists()
        assert art.meta["image_dim"] is None

    providers.text_embedder = None
    with (
        Pipeline(services.cfg, providers, db=services.db) as p,
        pytest.raises(EmbeddingsError, match="no embedder is configured"),
    ):
        p.run_stage("analysis.embeddings", a)


# --- the search ---------------------------------------------------------------------------------


def test_the_shot_described_by_the_query_comes_first_on_both_rankings(
    services: AppServices, asset: str
) -> None:
    result = services.search.search(asset, "雪山里的画面 sh_0003", limit=3)
    assert result.rankings == ["image", "text"]
    top = result.hits[0]
    assert top.shot_id.endswith("3") and set(top.similarity) == {"image", "text"}
    assert top.similarity["image"] > 0.99 and top.similarity["text"] > 0.5
    assert top.caption.startswith("雪山里的画面") and top.scene_id is not None
    assert top.thumbnail is not None and (services.cfg.data_dir / top.thumbnail).is_file()
    assert top.thumbnail.endswith("_b.jpg")
    assert [h.score for h in result.hits] == sorted((h.score for h in result.hits), reverse=True)
    assert len(result.hits) == 3


def test_credits_are_left_out_unless_asked_for_and_filters_narrow_the_search(
    services: AppServices, asset: str
) -> None:
    from offscreen.providers.ports import ShotFilter

    q = "雪山里的画面 sh_0004"
    hidden = services.search.search(asset, q, limit=10)
    assert [h.shot_id[-1] for h in hidden.hits].count("4") == 0 and len(hidden.hits) == 3
    shown = services.search.search(asset, q, limit=10, where=ShotFilter(exclude_credits=False))
    assert shown.hits[0].shot_id.endswith("4") and len(shown.hits) == 4

    late = services.search.search(asset, q, limit=10, where=ShotFilter(start_ms=8000))
    assert [h.start_ms for h in late.hits] and min(h.start_ms for h in late.hits) >= 8000
    scene = shown.hits[0].scene_id
    assert scene is not None
    assert len(services.search.search(asset, q, where=ShotFilter(scene_id=scene)).hits) == 3
    assert services.search.search(asset, q, where=ShotFilter(scene_id="sc_none")).hits == []


def test_the_derived_index_is_built_once_and_replaced_when_the_artifact_changes(
    services: AppServices, providers: Providers, asset: str
) -> None:
    services.search.search(asset, "雪山")
    root = services.cfg.data_dir / "index" / asset
    (built,) = list(root.iterdir())
    marker = built / ".complete"
    first = marker.stat().st_mtime_ns
    services.search.search(asset, "龙")
    assert marker.stat().st_mtime_ns == first  # reused

    other = FakeEmbedder(
        dim=256, image_of=picture_of, id="fake-clip@2"
    )  # a new model: new artifact
    providers.image_embedder = other
    with Pipeline(services.cfg, providers, db=services.db) as p:
        p.run_stage("analysis.embeddings", asset)
    services.search.search(asset, "雪山")
    (now,) = list(root.iterdir())  # the stale one is gone
    assert now != built


def test_bad_requests_and_missing_pieces(services: AppServices, movie: Path, asset: str) -> None:
    with pytest.raises(InvalidInput, match="query is empty"):
        services.search.search(asset, "  ")
    with pytest.raises(InvalidInput, match="limit must be between"):
        services.search.search(asset, "x", limit=0)
    with pytest.raises(NotFound, match="unknown asset"):
        services.search.search("ast_missing", "x")
    other = services.library.import_asset(
        str(movie.with_name("Other.mp4")) if False else str(movie)
    )
    assert other.id == asset  # the same file is the same asset
    fresh = movie.parent / "Fresh.mp4"
    fresh.write_bytes(movie.read_bytes() + b"\0")  # a different fingerprint
    new = services.library.import_asset(str(fresh))
    with pytest.raises(NotFound, match=r"analysis\.embeddings has not been built"):
        services.search.search(new.id, "x")


# --- HTTP ---------------------------------------------------------------------------------------


def test_the_search_route(client: TestClient, asset: str) -> None:
    r = client.get(
        f"/api/assets/{asset}/shots/search", params={"q": "雪山里的画面 sh_0002", "limit": 2}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "雪山里的画面 sh_0002" and body["rankings"] == ["image", "text"]
    assert len(body["hits"]) == 2 and body["hits"][0]["shot_id"].endswith("2")
    thumb = body["hits"][0]["thumbnail"]
    assert client.get(f"/api/files/{thumb}").status_code == 200

    both = client.get(
        f"/api/assets/{asset}/shots/search",
        params={"q": "雪山", "exclude_credits": "false", "limit": 10, "min_sharpness": 0},
    )
    assert len(both.json()["hits"]) == 4

    assert client.get(f"/api/assets/{asset}/shots/search").status_code == 422  # q is required
    assert (
        client.get(f"/api/assets/{asset}/shots/search", params={"q": "x", "limit": 101}).status_code
        == 422
    )
    missing = client.get("/api/assets/ast_missing/shots/search", params={"q": "x"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    blank = client.get(f"/api/assets/{asset}/shots/search", params={"q": "   "})
    assert blank.status_code == 422
