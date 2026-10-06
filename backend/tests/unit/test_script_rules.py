"""The script rule checker: each rule, the name-slip detector, and the annotations."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.script_rules import (
    MAX_SEGMENT_CHARS,
    MIN_SEGMENT_CHARS,
    KnownName,
    Rules,
    SegmentDraft,
    as_annotations,
    check,
    name_slips,
)

SCENES = ["sc_1", "sc_2"]


def seg(chars: int, refs: list[str] | None = None, id_: str | None = None) -> SegmentDraft:
    return SegmentDraft("字" * chars + "。", ["sc_1"] if refs is None else refs, id_)


def rules(**kw: object) -> Rules:
    return Rules(scene_ids=SCENES, **kw)  # type: ignore[arg-type]


def messages(segments: list[SegmentDraft], r: Rules) -> list[str]:
    return [v.message for v in check(segments, r)]


def test_the_limits_are_the_architecture_ones() -> None:
    assert (MIN_SEGMENT_CHARS, MAX_SEGMENT_CHARS) == (15, 80)


def test_a_good_text_has_no_violations() -> None:
    assert check([seg(40), seg(50, ["sc_2"])], rules(target_chars=90)) == []


def test_segment_length_bounds_are_inclusive() -> None:
    r = rules()
    assert check([seg(15), seg(80)], r) == []
    short, long_ = check([seg(14), seg(81)], r)
    assert (short.rule, short.segment) == ("segment_length", 0) and "太短" in short.message
    assert (long_.rule, long_.segment) == ("segment_length", 1) and "拆成多段" in long_.message
    assert check([seg(10)], rules(min_segment_chars=10)) == []  # a short beat may relax it


def test_scene_refs_must_exist_and_not_be_empty() -> None:
    found = check([seg(30, ["sc_9", "sc_1"]), seg(30, [])], rules())
    assert [(v.rule, v.segment) for v in found] == [("scene_refs", 0), ("scene_refs", 1)]
    assert "sc_9" in found[0].message and "sc_1" not in found[0].message
    assert "没有 scene_refs" in found[1].message


def test_banned_words_are_reported_per_segment() -> None:
    bad = SegmentDraft("字" * 20 + "不得不说，这很" + "字" * 5 + "精彩绝伦", ["sc_1"])
    found = check([seg(20), bad], rules(banned_words=["不得不说", "精彩绝伦", "没出现"]))
    assert [(v.rule, v.segment) for v in found] == [("banned_word", 1)]
    assert "不得不说, 精彩绝伦" in found[0].message


def test_total_length_has_inclusive_tolerance_and_says_which_way() -> None:
    def total(n: int) -> list[str]:
        return messages(
            [seg(n)], rules(target_chars=70, min_segment_chars=1, max_segment_chars=500)
        )

    assert total(60) == [] and total(80) == []  # 70 ± 15 % = 60..80 (rounded)
    assert "补充" in total(59)[0] and "目标 70 字（允许 60–80）" in total(59)[0]
    assert "删减" in total(81)[0]
    assert messages([seg(10)], rules(min_segment_chars=1)) == []  # no target: not checked
    assert messages([seg(30)], rules(target_chars=70, whole="本节"))[0].startswith("本节共")


def test_violations_come_segment_by_segment_then_the_total() -> None:
    found = check([seg(5, ["sc_9"]), seg(30)], rules(target_chars=200))
    assert [(v.rule, v.segment) for v in found] == [
        ("scene_refs", 0),
        ("segment_length", 0),
        ("total_length", None),
    ]


# --- names ----------------------------------------------------------------------------


SINTEL = KnownName("辛忒尔", ["小辛"])


def test_a_name_one_character_off_is_a_slip() -> None:
    assert name_slips("辛特尔走进了雪山。", [SINTEL]) == [("辛特尔", "辛忒尔")]
    assert name_slips("辛忒尔走进了雪山，辛特尔回头。", [SINTEL]) == [("辛特尔", "辛忒尔")]
    assert name_slips("辛忒尔和小辛是同一个人。", [SINTEL]) == []


def test_other_known_people_and_aliases_are_not_slips() -> None:
    cast = [SINTEL, KnownName("辛忒拉", [])]  # one character from the other's name
    assert name_slips("辛忒拉与辛忒尔相遇。", cast) == []
    assert name_slips("辛特拉出现了。", cast) == [("辛特拉", "辛忒拉")]
    assert name_slips("辛特尔来了", [KnownName("辛忒尔", ["辛特尔"])]) == []  # an alias


def test_short_names_are_not_checked() -> None:
    assert name_slips("小红来了，小龙走了。", [KnownName("小龙", [])]) == []


def test_latin_names_ignore_case_but_not_spelling() -> None:
    sintel = [KnownName("Sintel", [])]
    assert name_slips("sintel and SINTEL walk. Sintal waits.", sintel) == [("Sintal", "Sintel")]


def test_a_slip_is_a_violation_with_the_names_in_the_message() -> None:
    found = check([SegmentDraft("辛特尔" + "字" * 20, ["sc_1"])], rules(names=[SINTEL]))
    assert [v.rule for v in found] == ["name"]
    assert "「辛忒尔」写成了「辛特尔」" in found[0].message


@given(st.text(alphabet="辛忒尔小的了，。走", max_size=40))
def test_text_with_only_the_exact_name_never_has_slips(filler: str) -> None:
    text = "辛忒尔".join([filler, filler])
    assert all(slip != "辛忒尔" for slip, _ in name_slips(text, [SINTEL]))


# --- annotations ----------------------------------------------------------------------


def test_violations_become_rule_annotations_on_their_segments() -> None:
    segments = [seg(5, id_="seg_01"), seg(30, id_="seg_02")]
    found = check(segments, rules(target_chars=200))
    notes = as_annotations(found, ["seg_01", "seg_02"])
    assert [(a.segment_id, a.type) for a in notes] == [
        ("seg_01", "rule"),
        ("seg_02", "rule"),  # the total has no segment: it goes on the last one
    ]
    assert "太短" in notes[0].message and "共 35 字" in notes[1].message


@pytest.mark.parametrize("n", [0, 1, 2])
def test_no_violations_no_annotations(n: int) -> None:
    ids = [f"seg_{i}" for i in range(n + 1)]
    assert as_annotations([], ids) == []
