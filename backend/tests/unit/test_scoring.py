"""Unit tests for shot scoring with constructed data."""

from __future__ import annotations

import pytest

from offscreen.algo.scoring import (
    ScoringWeights,
    ShotScore,
    score_candidates,
    score_shot,
)
from offscreen.domain.index import ShotCaption, ShotCast, ShotIndexEntry


@pytest.fixture
def shot_entries() -> dict[str, ShotIndexEntry]:
    """Sample shot metadata."""
    return {
        "sh_0000": ShotIndexEntry(
            shot_id="sh_0000", start_ms=0, end_ms=3000, sharpness=0.9, brightness=0.8
        ),
        "sh_0001": ShotIndexEntry(
            shot_id="sh_0001",
            start_ms=3000,
            end_ms=6000,
            sharpness=0.7,
            brightness=0.5,
            is_credits=True,
        ),
        "sh_0002": ShotIndexEntry(
            shot_id="sh_0002",
            start_ms=6000,
            end_ms=9000,
            sharpness=0.8,
            brightness=0.85,
            caption="A wide landscape shot",
        ),
        "sh_0003": ShotIndexEntry(
            shot_id="sh_0003",
            start_ms=9000,
            end_ms=12000,
            sharpness=0.6,
            brightness=0.4,
            caption="A close-up of a person",
        ),
    }


@pytest.fixture
def shot_captions() -> dict[str, ShotCaption]:
    """LLM-generated shot descriptions."""
    return {
        "sh_0000": ShotCaption(shot_id="sh_0000", caption="A beautiful landscape with mountains"),
        "sh_0002": ShotCaption(
            shot_id="sh_0002",
            caption="Wide landscape showing hills and sky",
            action="panning left",
            emotion="peaceful",
        ),
        "sh_0003": ShotCaption(
            shot_id="sh_0003", caption="Close-up of a face showing emotion", has_onscreen_text=True
        ),
    }


def test_score_shot_basic(
    shot_entries: dict[str, ShotIndexEntry], shot_captions: dict[str, ShotCaption]
) -> None:
    """Score a single shot with default weights."""
    score = score_shot(
        shot_id="sh_0000",
        shot_entry=shot_entries["sh_0000"],
        shot_cast=None,
        shot_caption=shot_captions["sh_0000"],
        segment_text="A beautiful mountain landscape",
        embedding_scores={"sh_0000": 0.8},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    assert isinstance(score, ShotScore)
    assert score.shot_id == "sh_0000"
    assert 0.0 <= score.total <= 1.0
    assert score.embedding > 0.0
    assert score.caption > 0.0
    assert score.quality > 0.0


def test_score_shot_with_character_match() -> None:
    """Score is higher when mentioned characters appear."""
    entry = ShotIndexEntry(
        shot_id="sh_test", start_ms=0, end_ms=3000, sharpness=0.8, brightness=0.8
    )
    caption = ShotCaption(shot_id="sh_test", caption="Alice and Bob talk")

    from offscreen.domain.index import CastMember, ShotCast

    cast = ShotCast(
        shot_id="sh_test",
        characters=[CastMember(character_id="ch_alice", share=0.8, area_ratio=0.6)],
    )

    # Without character match
    score_no_char = score_shot(
        shot_id="sh_test",
        shot_entry=entry,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Alice speaks",
        embedding_scores={"sh_test": 0.5},
        character_ids={"ch_alice"},
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    # With character match
    score_with_char = score_shot(
        shot_id="sh_test",
        shot_entry=entry,
        shot_cast=cast,
        shot_caption=caption,
        segment_text="Alice speaks",
        embedding_scores={"sh_test": 0.5},
        character_ids={"ch_alice"},
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    assert score_with_char.character > score_no_char.character


def test_score_shot_reuse_penalty() -> None:
    """Shots in used_shot_ids get a reuse penalty."""
    entry = ShotIndexEntry(
        shot_id="sh_test", start_ms=0, end_ms=3000, sharpness=0.9, brightness=0.9
    )
    caption = ShotCaption(shot_id="sh_test", caption="A scene")

    score_new = score_shot(
        shot_id="sh_test",
        shot_entry=entry,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Some text",
        embedding_scores={"sh_test": 0.5},
        character_ids=set(),
        used_shot_ids=set(),  # not used yet
        segment_start_ms=0,
    )

    score_reused = score_shot(
        shot_id="sh_test",
        shot_entry=entry,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Some text",
        embedding_scores={"sh_test": 0.5},
        character_ids=set(),
        used_shot_ids={"sh_test"},  # already used
        segment_start_ms=0,
    )

    assert score_new.reuse > score_reused.reuse


def test_score_shot_time_order() -> None:
    """Forward chronological shots score higher."""
    entry_forward = ShotIndexEntry(
        shot_id="sh_forward", start_ms=5000, end_ms=8000, sharpness=0.8, brightness=0.8
    )
    entry_backward = ShotIndexEntry(
        shot_id="sh_backward", start_ms=1000, end_ms=4000, sharpness=0.8, brightness=0.8
    )

    caption = ShotCaption(shot_id="sh_test", caption="A scene")
    segment_start_ms = 4000

    score_forward = score_shot(
        shot_id="sh_forward",
        shot_entry=entry_forward,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Some text",
        embedding_scores={"sh_forward": 0.5},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=segment_start_ms,
    )

    score_backward = score_shot(
        shot_id="sh_backward",
        shot_entry=entry_backward,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Some text",
        embedding_scores={"sh_backward": 0.5},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=segment_start_ms,
    )

    assert score_forward.time_order > score_backward.time_order


def test_score_shot_exclusions() -> None:
    """Credits and on-screen text reduce scores."""
    entry_normal = ShotIndexEntry(
        shot_id="sh_normal", start_ms=0, end_ms=3000, sharpness=0.8, brightness=0.8
    )
    entry_credits = ShotIndexEntry(
        shot_id="sh_credits",
        start_ms=0,
        end_ms=3000,
        sharpness=0.8,
        brightness=0.8,
        is_credits=True,
    )

    caption_normal = ShotCaption(shot_id="sh_normal", caption="A scene")
    caption_text = ShotCaption(shot_id="sh_text", caption="A scene", has_onscreen_text=True)

    score_normal = score_shot(
        shot_id="sh_normal",
        shot_entry=entry_normal,
        shot_cast=None,
        shot_caption=caption_normal,
        segment_text="Some text",
        embedding_scores={"sh_normal": 0.5},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    score_credits = score_shot(
        shot_id="sh_credits",
        shot_entry=entry_credits,
        shot_cast=None,
        shot_caption=caption_normal,
        segment_text="Some text",
        embedding_scores={"sh_credits": 0.5},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    score_onscreen_text = score_shot(
        shot_id="sh_text",
        shot_entry=entry_normal,
        shot_cast=None,
        shot_caption=caption_text,
        segment_text="Some text",
        embedding_scores={"sh_text": 0.5},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    assert score_normal.exclusions > score_credits.exclusions
    assert score_normal.exclusions > score_onscreen_text.exclusions


def test_score_candidates_sorts_by_total(
    shot_entries: dict[str, ShotIndexEntry], shot_captions: dict[str, ShotCaption]
) -> None:
    """Candidate scoring returns sorted list."""
    shots_dict: dict[str, ShotCast] = {}
    embedding_scores = {
        "sh_0000": 0.9,
        "sh_0001": 0.3,
        "sh_0002": 0.8,
        "sh_0003": 0.6,
    }

    results = score_candidates(
        shot_ids=["sh_0000", "sh_0001", "sh_0002", "sh_0003"],
        shot_entries=shot_entries,
        shot_cast=shots_dict,
        shot_captions=shot_captions,
        segment_text="A beautiful landscape with mountains",
        embedding_scores=embedding_scores,
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
    )

    # Results should be sorted by total score (descending)
    assert len(results) > 0
    for i in range(len(results) - 1):
        assert results[i].total >= results[i + 1].total


def test_scoring_weights_validation() -> None:
    """Weights must sum to 1.0."""
    # Valid weights
    weights_ok = ScoringWeights(
        embedding=0.25,
        caption=0.20,
        character=0.15,
        quality=0.10,
        size=0.05,
        reuse=0.10,
        time_order=0.10,
        exclusions=0.05,
    )
    assert weights_ok  # should not raise

    # Invalid weights (sum too high)
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ScoringWeights(
            embedding=0.5,
            caption=0.5,
            character=0.5,
            quality=0.0,
            size=0.0,
            reuse=0.0,
            time_order=0.0,
            exclusions=0.0,
        )


def test_custom_weights_apply() -> None:
    """Custom weights change score distribution."""
    entry = ShotIndexEntry(
        shot_id="sh_test", start_ms=0, end_ms=3000, sharpness=0.8, brightness=0.8
    )
    caption = ShotCaption(shot_id="sh_test", caption="A scene")

    # All weight on embedding
    weights_embedding = ScoringWeights(
        embedding=1.0,
        caption=0.0,
        character=0.0,
        quality=0.0,
        size=0.0,
        reuse=0.0,
        time_order=0.0,
        exclusions=0.0,
    )

    score_embedding = score_shot(
        shot_id="sh_test",
        shot_entry=entry,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Some text",
        embedding_scores={"sh_test": 0.7},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
        weights=weights_embedding,
    )

    # All weight on quality
    weights_quality = ScoringWeights(
        embedding=0.0,
        caption=0.0,
        character=0.0,
        quality=1.0,
        size=0.0,
        reuse=0.0,
        time_order=0.0,
        exclusions=0.0,
    )

    score_quality = score_shot(
        shot_id="sh_test",
        shot_entry=entry,
        shot_cast=None,
        shot_caption=caption,
        segment_text="Some text",
        embedding_scores={"sh_test": 0.7},
        character_ids=set(),
        used_shot_ids=set(),
        segment_start_ms=0,
        weights=weights_quality,
    )

    # Embedding-weighted should be close to 0.7
    assert abs(score_embedding.total - 0.7) < 0.01
    # Quality-weighted should be close to average(0.8, 0.8) = 0.8
    assert abs(score_quality.total - 0.8) < 0.01
