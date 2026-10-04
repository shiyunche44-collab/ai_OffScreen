from __future__ import annotations

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.script import (
    build_outline,
    check_draft,
    count_chars,
    estimate_duration_s,
    target_chars,
)
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
