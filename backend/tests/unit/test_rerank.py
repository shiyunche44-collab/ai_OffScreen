"""Unit tests for LLM-based reranking."""

from __future__ import annotations

from offscreen.algo.rerank import RerankResult, ShotChoice, build_rerank_prompt
from offscreen.algo.scoring import ShotScore
from offscreen.domain.index import ShotCaption


def test_build_rerank_prompt_basic() -> None:
    """Prompt builder includes segment text and candidate descriptions."""
    segment_text = "A brave hero climbs the mountain"
    scored_shots = [
        ShotScore(
            shot_id="sh_0001",
            total=0.85,
            embedding=0.9,
            caption=0.8,
            character=0.0,
            quality=0.8,
            size=1.0,
            reuse=1.0,
            time_order=1.0,
            exclusions=1.0,
        ),
        ShotScore(
            shot_id="sh_0002",
            total=0.75,
            embedding=0.7,
            caption=0.75,
            character=0.0,
            quality=0.8,
            size=1.0,
            reuse=1.0,
            time_order=1.0,
            exclusions=1.0,
        ),
    ]
    shot_captions = {
        "sh_0001": ShotCaption(shot_id="sh_0001", caption="A wide shot of a mountain landscape"),
        "sh_0002": ShotCaption(shot_id="sh_0002", caption="A person climbing rocky terrain"),
    }

    prompt = build_rerank_prompt(segment_text, scored_shots, shot_captions, target_count=3)

    assert "A brave hero climbs the mountain" in prompt
    assert "sh_0001" in prompt
    assert "sh_0002" in prompt
    assert "A wide shot of a mountain landscape" in prompt
    assert "A person climbing rocky terrain" in prompt
    assert "JSON" in prompt


def test_build_rerank_prompt_truncates_to_top_10() -> None:
    """Prompt only includes top 10 shots to keep it reasonable."""
    segment_text = "Test"
    scored_shots = [
        ShotScore(
            shot_id=f"sh_{i:04d}",
            total=1.0 - i * 0.01,
            embedding=0.5,
            caption=0.5,
            character=0.0,
            quality=0.5,
            size=1.0,
            reuse=1.0,
            time_order=1.0,
            exclusions=1.0,
        )
        for i in range(20)
    ]
    shot_captions = {
        s.shot_id: ShotCaption(shot_id=s.shot_id, caption="Test") for s in scored_shots
    }

    prompt = build_rerank_prompt(segment_text, scored_shots, shot_captions)

    # Top 10 should be included
    assert "sh_0000" in prompt
    assert "sh_0009" in prompt
    # Beyond top 10 should not be included
    assert "sh_0010" not in prompt
    assert "sh_0019" not in prompt


def test_rerank_result_structure() -> None:
    """RerankResult and ShotChoice structure is valid."""
    choices = [
        ShotChoice(rank=1, shot_id="sh_001", reason="Wide shot of mountain"),
        ShotChoice(rank=2, shot_id="sh_002", reason="Close-up of climber"),
    ]
    result = RerankResult(choices=choices, summary="Selected shots show the climb progression")

    assert len(result.choices) == 2
    assert result.choices[0].rank == 1
    assert result.choices[1].shot_id == "sh_002"
    assert "climb" in result.summary.lower()
