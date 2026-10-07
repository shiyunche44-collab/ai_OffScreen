"""Unit tests for candidate shot generation."""

from __future__ import annotations

import pytest

from offscreen.algo.candidates import (
    CandidateSet,
    _shots_from_scenes,
    _shots_with_characters,
    candidates_for_segment,
)
from offscreen.domain.index import (
    Cast,
    CastMember,
    Character,
    Characters,
    Shot,
    ShotCaption,
    ShotCast,
    Shots,
)
from offscreen.domain.script import ScriptSegment


@pytest.fixture
def shots() -> Shots:
    """10 shots of 3 seconds each."""
    return Shots(
        asset_id="ast_test",
        shots=[Shot(id=f"sh_{i:04d}", start_ms=i * 3000, end_ms=(i + 1) * 3000) for i in range(10)],
    )


@pytest.fixture
def scenes_by_id() -> dict[str, list[str]]:
    """Two scenes: sc_001 = shots 0-2, sc_002 = shots 3-9."""
    return {
        "sc_001": ["sh_0000", "sh_0001", "sh_0002"],
        "sc_002": ["sh_0003", "sh_0004", "sh_0005", "sh_0006", "sh_0007", "sh_0008", "sh_0009"],
    }


@pytest.fixture
def shot_captions() -> dict[str, ShotCaption]:
    """Sample descriptions for searching."""
    return {
        f"sh_{i:04d}": ShotCaption(
            shot_id=f"sh_{i:04d}",
            caption=f"A scene with element {i}",
            action=None,
            emotion=None,
        )
        for i in range(10)
    }


@pytest.fixture
def characters() -> Characters:
    """Character registry."""
    return Characters(
        asset_id="ast_test",
        characters=[
            Character(id="ch_alice", name="Alice", name_source="ai"),
            Character(id="ch_bob", name="Bob", name_source="ai"),
        ],
    )


@pytest.fixture
def cast() -> Cast:
    """Shot-to-character mappings."""
    # Alice in shots 0-3, Bob in shots 4-7
    shots: list[ShotCast] = []
    for i in range(10):
        if i < 4:
            shots.append(
                ShotCast(
                    shot_id=f"sh_{i:04d}",
                    characters=[CastMember(character_id="ch_alice", share=0.8, area_ratio=0.6)],
                )
            )
        elif i < 8:
            shots.append(
                ShotCast(
                    shot_id=f"sh_{i:04d}",
                    characters=[CastMember(character_id="ch_bob", share=0.7, area_ratio=0.5)],
                )
            )
        else:
            shots.append(ShotCast(shot_id=f"sh_{i:04d}", characters=[]))
    return Cast(asset_id="ast_test", shots=shots)


def test_shots_from_scenes_collects_in_order(scenes_by_id: dict[str, list[str]]) -> None:
    shots = _shots_from_scenes(["sc_001"], scenes_by_id)
    assert shots == ["sh_0000", "sh_0001", "sh_0002"]


def test_shots_from_scenes_multiple_dedupes(scenes_by_id: dict[str, list[str]]) -> None:
    shots = _shots_from_scenes(["sc_001", "sc_002"], scenes_by_id)
    assert len(shots) == 10
    assert shots == [
        "sh_0000",
        "sh_0001",
        "sh_0002",
        "sh_0003",
        "sh_0004",
        "sh_0005",
        "sh_0006",
        "sh_0007",
        "sh_0008",
        "sh_0009",
    ]


def test_shots_from_scenes_unknown_scene(scenes_by_id: dict[str, list[str]]) -> None:
    shots = _shots_from_scenes(["sc_404"], scenes_by_id)
    assert shots == []


def test_shots_with_characters_finds_named_character(cast: Cast, characters: Characters) -> None:
    # Text mentions "Alice"
    shots = _shots_with_characters("A scene with Alice", cast, characters)
    # Alice is in shots 0-3, ranked by share (all 0.8), so chronologically
    assert shots == ["sh_0000", "sh_0001", "sh_0002", "sh_0003"]


def test_shots_with_characters_finds_multiple(cast: Cast, characters: Characters) -> None:
    # Text mentions both Alice and Bob
    shots = _shots_with_characters("Alice and Bob meet", cast, characters)
    # Alice shots (0-3, share 0.8) ranked first, then Bob shots (4-7, share 0.7)
    assert shots[:4] == ["sh_0000", "sh_0001", "sh_0002", "sh_0003"]
    assert shots[4:8] == ["sh_0004", "sh_0005", "sh_0006", "sh_0007"]


def test_shots_with_characters_case_insensitive(cast: Cast, characters: Characters) -> None:
    shots = _shots_with_characters("Meeting with ALICE", cast, characters)
    assert len(shots) == 4


def test_shots_with_characters_empty_text(cast: Cast, characters: Characters) -> None:
    shots = _shots_with_characters("", cast, characters)
    assert shots == []


def test_shots_with_characters_no_cast(characters: Characters) -> None:
    empty_cast = Cast(asset_id="ast_test", shots=[])
    shots = _shots_with_characters("Alice", empty_cast, characters)
    assert shots == []


def test_candidates_for_segment_fuses_sources(
    shots: Shots,
    scenes_by_id: dict[str, list[str]],
    shot_captions: dict[str, ShotCaption],
    cast: Cast,
    characters: Characters,
) -> None:
    segment = ScriptSegment(
        id="seg_01", kind="narration", beat="b", text="Alice and the dragon", scene_refs=["sc_001"]
    )

    result = candidates_for_segment(
        segment=segment,
        shots=shots,
        scenes_by_id=scenes_by_id,
        shot_captions=shot_captions,
        cast=cast,
        characters=characters,
    )

    assert isinstance(result, CandidateSet)
    assert len(result.shot_ids) > 0
    assert result.shot_ids[0] in result.sources  # at least one source tracked
    assert len(result.scores) == len(result.shot_ids)
    # Alice's shots (0-3) should rank high
    assert result.shot_ids[0] in ["sh_0000", "sh_0001", "sh_0002", "sh_0003"]


def test_candidates_for_segment_character_mention_only(
    shots: Shots,
    shot_captions: dict[str, ShotCaption],
    cast: Cast,
    characters: Characters,
) -> None:
    """When scene_refs are minimal but text has named characters, rank character shots highly."""
    segment = ScriptSegment(
        id="seg_01", kind="narration", beat="b", text="Unknown things happen", scene_refs=["sc_001"]
    )

    result = candidates_for_segment(
        segment=segment,
        shots=shots,
        scenes_by_id={"sc_001": ["sh_0000"]},
        shot_captions=shot_captions,
        cast=cast,
        characters=characters,
    )

    # Scene shot (sh_0000) and no character mentions -> minimal fusion
    assert len(result.shot_ids) > 0
    assert "sh_0000" in result.shot_ids


def test_candidates_for_segment_with_vector_search(
    shots: Shots,
    scenes_by_id: dict[str, list[str]],
    shot_captions: dict[str, ShotCaption],
    cast: Cast,
    characters: Characters,
) -> None:
    """Candidates incorporate vector search results."""
    segment = ScriptSegment(
        id="seg_01", kind="narration", beat="b", text="Beautiful scenery", scene_refs=["sc_001"]
    )

    search_results = {
        "embedding": ["sh_0009", "sh_0008", "sh_0007"],  # Top results from embedding
    }

    result = candidates_for_segment(
        segment=segment,
        shots=shots,
        scenes_by_id=scenes_by_id,
        shot_captions=shot_captions,
        cast=cast,
        characters=characters,
        search_results=search_results,
    )

    # Scene shots (0-2) and embedding shots (7-9) are fused
    assert len(result.shot_ids) > 0
    # Both sources should be represented
    sources_used = {source for sources in result.sources.values() for source in sources}
    assert "scenes" in sources_used
    assert "embedding" in sources_used
