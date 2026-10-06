"""Searching shots: rank fusion, the LanceDB index, the sentence-transformers adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.search import fuse, search_text
from offscreen.domain.index import ShotCaption
from offscreen.providers.adapters.fake import FakeEmbedder
from offscreen.providers.adapters.lancedb_index import LanceShotIndex, where_clause
from offscreen.providers.adapters.st_embedder import SentenceTransformerEmbedder
from offscreen.providers.ports import IndexedShot, ShotFilter

# --- algo ---------------------------------------------------------------------------------


def test_search_text_joins_what_is_seen_with_action_and_mood() -> None:
    cap = ShotCaption(shot_id="sh_1", caption="女孩抱着小龙。", action="抱起", emotion="悲伤")
    assert search_text(cap) == "女孩抱着小龙。动作：抱起。情绪：悲伤"
    assert search_text(ShotCaption(shot_id="sh_1", caption=" 雪山 ")) == "雪山"
    blank = ShotCaption(shot_id="sh_1", caption="x", action="  ", emotion=None)
    assert search_text(blank) == "x"


def test_fuse_rewards_shots_that_rank_well_in_several_lists() -> None:
    merged = fuse({"image": ["a", "b", "c"], "text": ["b", "d", "a"]})
    ids = [i for i, _ in merged]
    assert ids[:2] == ["b", "a"]  # b is 2nd and 1st; a is 1st and 3rd; the others appear once
    assert set(ids) == {"a", "b", "c", "d"}
    assert fuse({}) == [] and fuse({"x": []}) == []
    assert [i for i, _ in fuse({"only": ["x", "y"]})] == ["x", "y"]


def test_fuse_ties_go_to_the_earlier_list_and_rank() -> None:
    # a and d each appear once at rank 1 in different lists: the first list wins
    assert [i for i, _ in fuse({"one": ["a"], "two": ["d"]})] == ["a", "d"]


@given(
    st.lists(st.lists(st.sampled_from("abcdef"), unique=True, max_size=6), min_size=1, max_size=3)
)
def test_fuse_returns_every_shot_once_best_first(lists: list[list[str]]) -> None:
    out = fuse({str(i): ids for i, ids in enumerate(lists)})
    ids = [i for i, _ in out]
    assert len(ids) == len(set(ids)) == len({x for ids in lists for x in ids})
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


# --- the LanceDB index ----------------------------------------------------------------------


def unit(*v: float) -> list[float]:
    a = np.asarray(v, dtype=float)
    return [float(x) for x in a / np.linalg.norm(a)]


def shot(i: int, image: list[float] | None, text: list[float] | None, **over: Any) -> IndexedShot:
    base: dict[str, Any] = dict(
        shot_id=f"sh_{i}",
        scene_id="sc_1" if i < 3 else "sc_2",
        start_ms=i * 1000,
        end_ms=(i + 1) * 1000,
        sharpness=0.2 * i,
        brightness=0.5,
        is_credits=False,
    )
    base.update(over)
    return IndexedShot(image=image, text=text, **base)


@pytest.fixture
def index(tmp_path: Path) -> tuple[LanceShotIndex, Path]:
    shots = [
        shot(0, unit(1, 0, 0), unit(0, 1, 0)),
        shot(1, unit(0.9, 0.1, 0), unit(0, 0.9, 0.1)),
        shot(2, unit(0, 1, 0), unit(1, 0, 0)),
        shot(3, unit(0, 0, 1), unit(0, 0, 1), is_credits=True),
        shot(4, unit(0.7, 0.7, 0), unit(0.7, 0, 0.7), scene_id="it's a scene"),
    ]
    path = tmp_path / "idx"
    idx = LanceShotIndex()
    idx.build(path, shots)
    return idx, path


def ids(hits: list[Any]) -> list[str]:
    return [h.shot_id for h in hits]


def test_nearest_shots_come_first_with_their_cosine_similarity(index: Any) -> None:
    idx, path = index
    hits = idx.search(path, "image", unit(1, 0, 0), limit=3, where=ShotFilter())
    assert ids(hits) == ["sh_0", "sh_1", "sh_4"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[1].similarity == pytest.approx(0.9939, abs=1e-3)
    assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)
    # the text column is a different space
    assert ids(idx.search(path, "text", unit(1, 0, 0), limit=1, where=ShotFilter())) == ["sh_2"]


def test_filters_apply_before_the_ranking_and_credits_are_hidden_by_default(index: Any) -> None:
    idx, path = index
    q = unit(0, 0, 1)
    assert "sh_3" not in ids(idx.search(path, "image", q, limit=10, where=ShotFilter()))
    with_credits = ShotFilter(exclude_credits=False)
    assert ids(idx.search(path, "image", q, limit=1, where=with_credits)) == ["sh_3"]

    in_scene = ShotFilter(scene_id="sc_1")
    assert set(ids(idx.search(path, "image", q, limit=10, where=in_scene))) == {
        "sh_0",
        "sh_1",
        "sh_2",
    }
    window = ShotFilter(start_ms=1000, end_ms=3000)
    assert set(ids(idx.search(path, "image", q, limit=10, where=window))) == {"sh_1", "sh_2"}
    sharp = ShotFilter(min_sharpness=0.5)
    assert set(ids(idx.search(path, "image", q, limit=10, where=sharp))) == {"sh_4"} | set()
    assert idx.search(path, "image", q, limit=5, where=ShotFilter(min_brightness=0.9)) == []


def test_quotes_in_filter_values_cannot_break_out(index: Any) -> None:
    idx, path = index
    hits = idx.search(
        path, "image", unit(1, 1, 0), limit=5, where=ShotFilter(scene_id="it's a scene")
    )
    assert ids(hits) == ["sh_4"]
    evil = ShotFilter(scene_id="x' OR 1=1 --")
    assert idx.search(path, "image", unit(1, 0, 0), limit=5, where=evil) == []
    assert where_clause(ShotFilter(exclude_credits=False)) is None
    assert "''" in (where_clause(ShotFilter(scene_id="a'b")) or "")


def test_a_column_exists_only_when_every_shot_has_it(tmp_path: Path) -> None:
    idx = LanceShotIndex()
    path = tmp_path / "idx"
    idx.build(path, [shot(0, unit(1, 0), None), shot(1, unit(0, 1), None)])
    assert ids(idx.search(path, "image", unit(1, 0), limit=1, where=ShotFilter())) == ["sh_0"]
    with pytest.raises(LookupError, match="no text vectors"):
        idx.search(path, "text", unit(1, 0), limit=1, where=ShotFilter())
    idx.build(path, [shot(0, unit(1, 0), unit(1, 0)), shot(1, None, unit(0, 1))])  # rebuilt
    with pytest.raises(LookupError, match="no image vectors"):
        idx.search(path, "image", unit(1, 0), limit=1, where=ShotFilter())


def test_vectors_of_different_lengths_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different lengths"):
        LanceShotIndex().build(tmp_path / "i", [shot(0, [1.0, 0.0], None), shot(1, [1.0], None)])


# --- the sentence-transformers adapter --------------------------------------------------------


class StubModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.seen: list[Any] = []

    def encode(self, items: list[Any], **kw: Any) -> np.ndarray:
        assert kw["normalize_embeddings"] is True
        self.seen.append(items)
        return np.tile(np.arange(1, self.dim + 1, dtype=np.float32), (len(items), 1))


def test_joint_model_embeds_images_and_texts_with_their_own_side(tmp_path: Path) -> None:
    from PIL import Image

    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(img)
    pic, txt = StubModel(3), StubModel(3)
    emb = SentenceTransformerEmbedder(
        "multi-text", image_model="clip-img", models={"clip-img": pic, "multi-text": txt}
    )
    assert emb.id == "sentence-transformers/clip-img+multi-text@1"
    assert emb.embed_images([img, img]) == [[1.0, 2.0, 3.0]] * 2
    assert len(pic.seen[0]) == 2 and pic.seen[0][0].mode == "RGB" and txt.seen == []
    assert emb.embed_texts(["雪山", "龙"]) == [[1.0, 2.0, 3.0]] * 2
    assert txt.seen == [["雪山", "龙"]]
    assert emb.dim == 3
    assert emb.embed_texts([]) == [] and emb.embed_images([]) == []


def test_a_text_only_model_refuses_images_and_bad_files_are_named(tmp_path: Path) -> None:
    emb = SentenceTransformerEmbedder("bge", models={"bge": StubModel(2)})
    assert emb.id.startswith("sentence-transformers/-+bge")
    with pytest.raises(RuntimeError, match="text only"):
        emb.embed_images([tmp_path / "x.jpg"])

    joint = SentenceTransformerEmbedder("t", image_model="i", models={"i": StubModel(2)})
    bad = tmp_path / "not-an-image.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(RuntimeError, match="cannot read the image"):
        joint.embed_images([bad])


def test_a_missing_package_says_how_to_install_it() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
            SentenceTransformerEmbedder("bge").embed_texts(["x"])


# --- the fake used by other tests --------------------------------------------------------------


def test_the_fake_embedder_puts_similar_texts_close_together() -> None:
    fake = FakeEmbedder(dim=128)
    a, b, c = fake.embed_texts(["雪山里的龙在飞", "雪山里的龙", "市场上的人群"])
    dot = lambda x, y: sum(p * q for p, q in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c) and dot(a, a) == pytest.approx(1.0)
    assert fake.text_calls == [["雪山里的龙在飞", "雪山里的龙", "市场上的人群"]]
