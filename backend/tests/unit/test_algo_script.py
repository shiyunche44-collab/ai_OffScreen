from __future__ import annotations

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.script import (
    build_outline,
    check_beat,
    check_draft,
    count_chars,
    estimate_duration_s,
    key_lines,
    scene_lines,
    target_chars,
)
from offscreen.domain.index import Scene, TranscriptLine
from offscreen.domain.script import ScriptSegment


def seg(i: int, text: str, refs: list[str], beat: str | None = None) -> ScriptSegment:
    return ScriptSegment(id=f"seg_{i:02d}", kind="narration", beat=beat, text=text, scene_refs=refs)


def test_count_chars_ignores_punctuation_and_space() -> None:
    assert count_chars("这个女孩，为了一条龙！ OK 2 个") == 13
    assert count_chars("……  ，。") == 0


def test_target_and_estimate_round_trip() -> None:
    assert target_chars(180, 4.5) == 810
    assert estimate_duration_s(810, 4.5) == 180


def test_check_draft_ok() -> None:
    texts = ["一" * 50, "二" * 50]
    assert check_draft(texts, [["sc_1"], ["sc_2"]], ["sc_1", "sc_2"], target=100) == []


def test_check_draft_reports_each_problem() -> None:
    texts = ["一" * 50, "二" * 2, "三" * 250]
    errs = check_draft(texts, [["sc_1"], ["sc_9"], []], ["sc_1"], target=100)
    joined = "\n".join(errs)
    assert "sc_9" in joined  # unknown scene
    assert "第 3 段没有 scene_refs" in joined
    assert "第 2 段只有 2 字" in joined
    assert "第 3 段有 250 字" in joined
    assert "删减" in joined  # total 302 vs 100


def test_check_draft_too_short_overall() -> None:
    errs = check_draft(["一" * 30], [["sc_1"]], ["sc_1"], target=100)
    assert len(errs) == 1 and "补充" in errs[0]


def test_check_draft_tolerance_bounds_are_inclusive() -> None:
    assert check_draft(["一" * 85], [["a"]], ["a"], target=100) == []
    assert check_draft(["一" * 115], [["a"]], ["a"], target=100) == []
    assert check_draft(["一" * 116], [["a"]], ["a"], target=100) != []


def test_outline_groups_runs_and_dedupes_scenes() -> None:
    segs = [
        seg(1, "一" * 9, ["sc_1"], "hook"),
        seg(2, "二" * 18, ["sc_1", "sc_2"], "hook"),
        seg(3, "三" * 45, ["sc_3"], "climax"),
        seg(4, "四" * 9, ["sc_1"], None),
        seg(5, "五" * 9, ["sc_2"], None),
    ]
    out = build_outline(segs, 4.5)
    assert [(o.beat, o.scene_refs, o.target_s) for o in out] == [
        ("hook", ["sc_1", "sc_2"], 6),
        ("climax", ["sc_3"], 10),
        ("body", ["sc_1", "sc_2"], 4),
    ]


@given(
    st.lists(
        st.tuples(
            st.text(alphabet="甲乙丙丁，。 ", min_size=1, max_size=40).filter(
                lambda t: count_chars(t) > 0
            ),
            st.sampled_from(["hook", "setup", None]),
        ),
        min_size=1,
        max_size=15,
    )
)
def test_outline_covers_all_narration_time(items: list[tuple[str, str | None]]) -> None:
    segs = [seg(i, t, ["sc_1"], b) for i, (t, b) in enumerate(items, 1)]
    out = build_outline(segs, 4.5)
    assert sum(o.target_s for o in out) >= len(out)
    assert all(a.beat != b.beat for a, b in pairwise(out))


# --- per-beat checks and the key lines of a scene ---------------------------------------


def line(i: int, start: int, end: int, text: str) -> TranscriptLine:
    return TranscriptLine(id=f"ln_{i}", start_ms=start, end_ms=end, text=text)


def scene(start: int, end: int, line_ids: list[str] | None = None) -> Scene:
    return Scene(
        id="sc_1",
        start_ms=start,
        end_ms=end,
        shot_ids=["sh_1"],
        summary="x",
        line_ids=line_ids or [],
    )


def test_check_beat_reports_banned_words_and_speaks_of_this_section() -> None:
    errors = check_beat(
        ["字" * 20 + "不得不说"], [["sc_1"]], ["sc_1"], target=45, banned_words=["不得不说", "别的"]
    )
    assert any("禁用词：不得不说" in e for e in errors)
    assert any(e.startswith("本节共") for e in errors)
    assert not any("别的" in e for e in errors)  # only the words that occur


def test_check_beat_accepts_a_good_beat() -> None:
    assert (
        check_beat(["字" * 44 + "。"], [["sc_1"]], ["sc_1"], target=45, banned_words=["坏"]) == []
    )


def test_scene_lines_prefer_the_scenes_own_line_list_else_time_overlap() -> None:
    lines = [line(1, 0, 1000, "a"), line(2, 5000, 6000, "b"), line(3, 9000, 9500, "c")]
    assert [x.id for x in scene_lines(scene(4000, 9200, ["ln_1"]), lines)] == ["ln_1"]
    assert [x.id for x in scene_lines(scene(4000, 9200), lines)] == ["ln_2", "ln_3"]
    assert [x.id for x in scene_lines(scene(6000, 9000), lines)] == []  # touching is not overlap


def test_key_lines_are_the_longest_in_time_order_and_cut() -> None:
    lines = [
        line(1, 0, 1, "嗯"),
        line(2, 1, 2, "我们不能留在这里，他们很快就会找到我们。"),
        line(3, 2, 3, "好"),
        line(4, 3, 4, "再见了，小龙。"),
        line(5, 4, 5, "走！"),
    ]
    assert key_lines(lines, limit=2) == [
        "我们不能留在这里，他们很快就会找到我们。",
        "再见了，小龙。",
    ]
    cut = key_lines([line(1, 0, 1, "字" * 100)], max_chars=10)
    assert cut == ["字" * 9 + "…"]
    assert key_lines([]) == []
    assert key_lines([line(1, 0, 1, "  ")]) == []


@given(
    st.lists(
        st.text(alphabet=st.characters(categories=["Lo"]), min_size=1, max_size=30),
        unique=True,
        max_size=12,
    ),
    st.integers(1, 5),
)
def test_key_lines_never_exceed_the_limit_and_keep_time_order(texts: list[str], limit: int) -> None:
    lines = [line(i, i, i + 1, t) for i, t in enumerate(texts)]
    picked = key_lines(lines, limit=limit)
    assert len(picked) <= limit
    order = [" ".join(t.split()) for t in texts]
    positions = [order.index(p) if p in order else -1 for p in picked if not p.endswith("…")]
    assert positions == sorted(positions)
