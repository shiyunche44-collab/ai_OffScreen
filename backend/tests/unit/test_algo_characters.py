"""Name checks and the human revision layer for characters."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.characters import (
    apply_overrides,
    check_names,
    match_centres,
    remap_overrides,
    resolve_merges,
)
from offscreen.domain.index import Character, CharacterOverride, FaceCluster

LINES = ["Where are you going?", "Sintel,   come back!", "To the mountain.\nThe dragon is there."]


def test_a_name_needs_evidence_that_is_a_line_of_the_film() -> None:
    ids = ["ch_01", "ch_02", "ch_03", "ch_04"]
    found = check_names(
        {
            "ch_01": ("Sintel", "sintel, come back!"),  # case and spacing do not matter
            "ch_02": ("Scales", None),  # no evidence
            "ch_03": ("Gunnar", "Gunnar is here"),  # invented quote
            "ch_04": (None, None),  # unnamed is fine
        },
        ids,
        LINES,
    )
    assert set(found) == {"ch_02", "ch_03"}
    assert "没有给出台词依据" in found["ch_02"][0]
    assert "不是台词里的原句" in found["ch_03"][0]


def test_a_quote_may_cover_part_of_a_line_or_run_across_two() -> None:
    assert check_names({"ch_01": ("X", "mountain. The dragon")}, ["ch_01"], LINES) == {}
    assert check_names({"ch_01": ("X", "going? Sintel")}, ["ch_01"], LINES) == {}


def test_skipped_and_invented_ids_are_problems() -> None:
    found = check_names({"ch_09": ("X", "Sintel")}, ["ch_01"], LINES)
    assert "没有输出" in found["ch_01"][0] and "不在人物列表" in found["ch_09"][0]


# --- re-finding people after a re-clustering ------------------------------------------------


def people(n: int, seed: int = 0) -> np.ndarray:
    m = np.random.default_rng(seed).normal(size=(n, 16))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def vec(v: np.ndarray) -> list[float]:
    return [float(x) for x in v]


def test_match_centres_pairs_the_same_people_one_to_one() -> None:
    a, b, c = people(3)
    jitter = np.random.default_rng(1).normal(size=16) * 0.02
    got = match_centres(
        {"old_a": vec(a), "old_b": vec(b)},
        {"new_1": vec(b + jitter), "new_2": vec(a + jitter), "new_3": vec(c)},
    )
    assert got == {"old_a": "new_2", "old_b": "new_1"}
    # two wanted centres for one current centre: only the closer gets it
    both = match_centres({"x": vec(a), "y": vec(a + 5 * jitter)}, {"n": vec(a)})
    assert both == {"x": "n"}
    assert match_centres({"x": vec(a)}, {"n": vec(c)}) == {}
    assert match_centres({}, {"n": vec(c)}) == {} and match_centres({"x": vec(a)}, {}) == {}
    assert match_centres({"x": [1.0, 0.0]}, {"n": [1.0, 0.0, 0.0]}) == {}  # other model: no match


@given(st.integers(0, 40))
def test_match_centres_never_reuses_a_current_centre(seed: int) -> None:
    wanted = {f"w{i}": vec(v) for i, v in enumerate(people(5, seed))}
    current = {f"c{i}": vec(v) for i, v in enumerate(people(4, seed + 100))}
    got = match_centres(wanted, current, threshold=-1.0)
    assert len(set(got.values())) == len(got) == 4


def test_edits_follow_their_person_to_a_new_id_and_orphans_are_kept_aside() -> None:
    a, b, c = people(3, 5)
    edits = [
        CharacterOverride(character_id="ch_01", name="Sintel", centroid=vec(a)),
        CharacterOverride(character_id="ch_02", name="Gone", centroid=vec(c)),
        CharacterOverride(
            character_id="ch_03", merged_into="ch_01", centroid=vec(b), merged_into_centroid=vec(a)
        ),
    ]
    # a new clustering: the people changed places and c is no longer found
    current = {"ch_01": vec(b), "ch_02": vec(a)}
    remapped, orphans = remap_overrides(edits, current)
    by_id = {o.character_id: o for o in remapped}
    assert by_id["ch_02"].name == "Sintel"  # a is ch_02 now
    assert by_id["ch_01"].merged_into == "ch_02"  # b joined a
    assert [o.name for o in orphans] == ["Gone"]


def test_edits_without_a_centre_keep_their_id_only_if_it_exists() -> None:
    kept = CharacterOverride(character_id="ch_01", name="A")
    lost = CharacterOverride(character_id="ch_07", name="B")
    remapped, orphans = remap_overrides([kept, lost], {"ch_01": [1.0, 0.0]})
    assert remapped == [kept] and orphans == [lost]


def test_two_old_edits_that_land_on_one_person_do_not_both_apply() -> None:
    a = people(1)[0]
    first = CharacterOverride(character_id="ch_01", name="Old", centroid=vec(a))
    second = CharacterOverride(character_id="ch_02", name="New", centroid=vec(a))
    remapped, orphans = remap_overrides([first, second], {"ch_05": vec(a)})
    assert [(o.character_id, o.name) for o in remapped] == [
        ("ch_05", "Old")
    ]  # one person, one edit
    assert [o.name for o in orphans] == ["New"]  # the other is kept aside, not lost


# --- applying edits --------------------------------------------------------------------------


def char(cid: str, size: int, name: str | None = None) -> Character:
    return Character(
        id=cid,
        name=name,
        name_source="ai" if name else None,
        aliases=["x"],
        face_cluster=FaceCluster(size=size, centroid_ref=f"c.npy#{cid[-1]}", thumbnails=[]),
    )


def test_human_names_win_ignored_are_hidden_merged_add_their_faces() -> None:
    chars = [char("ch_01", 10, "AI name"), char("ch_02", 6), char("ch_03", 4), char("ch_04", 2)]
    shown, merged, ignored = apply_overrides(
        chars,
        [
            CharacterOverride(character_id="ch_01", name="Sintel", aliases=["girl"]),
            CharacterOverride(character_id="ch_02", merged_into="ch_01"),
            CharacterOverride(character_id="ch_03", ignored=True),
        ],
    )
    assert [c.id for c in shown] == ["ch_01", "ch_04"]
    first = shown[0]
    assert (first.name, first.name_source, first.aliases) == ("Sintel", "human", ["girl"])
    assert first.face_cluster is not None and first.face_cluster.size == 16
    assert merged == {"ch_02": "ch_01"} and ignored == ["ch_03"]
    assert shown[1] == chars[3]  # untouched


def test_an_empty_name_says_the_character_has_none() -> None:
    shown, _, _ = apply_overrides(
        [char("ch_01", 3, "Wrong")], [CharacterOverride(character_id="ch_01", name="")]
    )
    assert shown[0].name is None and shown[0].name_source is None


def test_merge_chains_follow_through_and_cycles_or_ignored_targets_are_not_merges() -> None:
    chain = [
        CharacterOverride(character_id="ch_a", merged_into="ch_b"),
        CharacterOverride(character_id="ch_b", merged_into="ch_c"),
    ]
    assert resolve_merges(chain) == {"ch_a": "ch_c", "ch_b": "ch_c"}
    loop = [
        CharacterOverride(character_id="ch_a", merged_into="ch_b"),
        CharacterOverride(character_id="ch_b", merged_into="ch_a"),
    ]
    assert resolve_merges(loop) == {}
    onto_ignored = [
        CharacterOverride(character_id="ch_a", merged_into="ch_b"),
        CharacterOverride(character_id="ch_b", ignored=True),
    ]
    assert resolve_merges(onto_ignored) == {}
    assert resolve_merges([CharacterOverride(character_id="ch_a", merged_into="ch_a")]) == {}


def test_applying_edits_does_not_change_the_input() -> None:
    chars = [char("ch_01", 5)]
    apply_overrides(chars, [CharacterOverride(character_id="ch_01", name="N")])
    assert chars[0].name is None


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b"])
def test_overrides_store_refuses_odd_asset_ids(bad: str, tmp_path: pytest.TempPathFactory) -> None:
    from pathlib import Path

    from offscreen.store.overrides import OverridesStore

    with pytest.raises(ValueError, match="bad asset id"):
        OverridesStore(Path(str(tmp_path))).read_characters(bad)
