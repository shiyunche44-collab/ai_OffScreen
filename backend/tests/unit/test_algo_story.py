from __future__ import annotations

from offscreen.algo.story import check_story_refs, fmt_clock

IDS = ["sc_001", "sc_002", "sc_003", "sc_004"]


def test_fmt_clock() -> None:
    assert fmt_clock(0) == "00:00"
    assert fmt_clock(61_999) == "01:01"
    assert fmt_clock(3_725_000) == "1:02:05"


def test_valid_refs_have_no_problems() -> None:
    assert check_story_refs([["sc_001", "sc_002"], ["sc_003", "sc_004"]], ["sc_003"], IDS) == []


def test_unknown_scene_in_an_act_a_turning_point_or_the_ending() -> None:
    errors = check_story_refs([["sc_001", "sc_9"]], ["sc_8"], IDS, ending_ids=["sc_7"])
    assert errors == [
        "第 1 幕引用了不存在的场景 sc_9",
        "关键转折引用了不存在的场景 sc_8",
        "结局引用了不存在的场景 sc_7",
    ]


def test_a_scene_in_two_acts() -> None:
    errors = check_story_refs([["sc_001", "sc_002"], ["sc_002", "sc_003"]], [], IDS)
    assert errors == ["场景 sc_002 同时出现在第 1 幕和第 2 幕"]


def test_cover_demands_that_every_scene_belongs_to_an_act() -> None:
    acts = [["sc_001"], ["sc_003"]]
    assert check_story_refs(acts, [], IDS) == []  # coverage is opt-in
    assert check_story_refs(acts, [], IDS, cover=True) == ["这些场景不属于任何一幕：sc_002、sc_004"]


def test_a_long_list_of_lost_scenes_is_shortened() -> None:
    ids = [f"sc_{i:03d}" for i in range(20)]
    (msg,) = check_story_refs([[ids[0]]], [], ids, cover=True)
    assert msg.endswith("……") and "sc_009" not in msg
