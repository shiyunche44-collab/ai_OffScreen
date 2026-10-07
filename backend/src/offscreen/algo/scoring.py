"""Shot scoring for selection: multi-factor ranking with configurable weights.

Factors:
1. **Image-text similarity** (embedding): visual relevance to segment text
2. **Description similarity**: semantic match between shot caption and text
3. **Character match**: score boost if named characters appear in shot
4. **Image quality**: sharpness + brightness
5. **Shot size**: preference match (long shot, medium, close-up, etc.)
6. **Reuse penalty**: avoid repeating the same shot
7. **Time order**: prefer forward-chronological shots
8. **Exclusions**: penalize credits, on-screen text, black frames

All scores are normalized to [0, 1]; final score is a weighted sum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from offscreen.domain.index import ShotCaption, ShotCast, ShotIndexEntry


@dataclass(frozen=True)
class ScoringWeights:
    """Normalized weights for scoring factors. Must sum to 1.0."""

    embedding: float = 0.25  # vector search similarity
    caption: float = 0.20  # LLM shot description
    character: float = 0.15  # named character presence
    quality: float = 0.10  # sharpness + brightness
    size: float = 0.05  # shot size preference
    reuse: float = 0.10  # penalty for repeating shots
    time_order: float = 0.10  # chronological order preference
    exclusions: float = 0.05  # credits, text, black frames

    def __post_init__(self) -> None:
        # Check weights sum to ~1.0 (allow small float error)
        total = sum(
            [
                self.embedding,
                self.caption,
                self.character,
                self.quality,
                self.size,
                self.reuse,
                self.time_order,
                self.exclusions,
            ]
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"weights must sum to 1.0, got {total}")


@dataclass
class ShotScore:
    """Per-shot score breakdown for debugging."""

    shot_id: str
    total: float  # 0..1
    embedding: float
    caption: float
    character: float
    quality: float
    size: float
    reuse: float
    time_order: float
    exclusions: float
    reason: str = ""  # debug annotation


def score_shot(
    shot_id: str,
    shot_entry: ShotIndexEntry,
    shot_cast: ShotCast | None,
    shot_caption: ShotCaption | None,
    segment_text: str,
    embedding_scores: dict[str, float],
    character_ids: set[str],
    used_shot_ids: set[str],
    segment_start_ms: int,
    weights: ScoringWeights | None = None,
    size_preference: Literal["long", "medium", "close"] | None = None,
    caption_similarity: Callable[[str, str], float] | None = None,
) -> ShotScore:
    """Score a single shot against a segment.

    Args:
        shot_id: The shot to score
        shot_entry: Metadata (quality, credits, on-screen text, time)
        shot_cast: Who appears in this shot
        shot_caption: LLM-generated description
        segment_text: The narration text to match against
        embedding_scores: shot_id -> similarity from vector search
        character_ids: set of character IDs mentioned in segment text
        used_shot_ids: shots already used (for reuse penalty)
        segment_start_ms: start time of this segment (for time order)
        weights: scoring weights (default weights if None)
        size_preference: preferred shot size if any
        caption_similarity: function(caption_text, segment_text) -> [0, 1]
            (default: simple substring match)

    Returns:
        ShotScore with component scores and total
    """
    if weights is None:
        weights = ScoringWeights()

    # 1. Embedding similarity (vector search)
    embedding_sim = embedding_scores.get(shot_id, 0.0)

    # 2. Caption similarity (shot description vs segment text)
    if shot_caption and caption_similarity:
        caption_sim = caption_similarity(shot_caption.caption, segment_text)
    elif shot_caption:
        # Fallback: simple overlap scoring
        caption_text = shot_caption.caption.lower()
        segment_lower = segment_text.lower()
        words_in_segment = set(segment_lower.split())
        matching_words = sum(1 for w in caption_text.split() if w in words_in_segment)
        total_words = max(len(caption_text.split()), len(words_in_segment))
        caption_sim = matching_words / total_words if total_words > 0 else 0.0
    else:
        caption_sim = 0.0

    # 3. Character presence
    character_score = 0.0
    if shot_cast and character_ids:
        for member in shot_cast.characters:
            if member.character_id in character_ids:
                # Weight by share of screen time
                character_score = max(character_score, member.share)

    # 4. Image quality (sharpness + brightness, normalized)
    quality_score = (shot_entry.sharpness + shot_entry.brightness) / 2

    # 5. Shot size (if preference stated)
    size_score = 1.0  # neutral default
    if size_preference and shot_caption:
        # Simple heuristic: check caption for size keywords
        caption_lower = shot_caption.caption.lower()
        if (
            size_preference == "long" and ("wide" in caption_lower or "landscape" in caption_lower)
        ) or (
            size_preference == "close"
            and ("closeup" in caption_lower or "close-up" in caption_lower)
        ):
            size_score = 1.0
        else:
            size_score = 0.8  # mild penalty for non-preference

    # 6. Reuse penalty
    reuse_score = 0.0 if shot_id in used_shot_ids else 1.0

    # 7. Time order (prefer forward time flow)
    time_order_score = 1.0 if shot_entry.start_ms >= segment_start_ms else 0.5

    # 8. Exclusions (penalize credits, on-screen text, black frames)
    exclusions_score = 1.0
    if shot_entry.is_credits:
        exclusions_score *= 0.1
    if shot_caption and shot_caption.has_onscreen_text:
        exclusions_score *= 0.2
    # (black frame detection would require image analysis; skip for v1)

    # Compute weighted total
    total = (
        embedding_sim * weights.embedding
        + caption_sim * weights.caption
        + character_score * weights.character
        + quality_score * weights.quality
        + size_score * weights.size
        + reuse_score * weights.reuse
        + time_order_score * weights.time_order
        + exclusions_score * weights.exclusions
    )

    return ShotScore(
        shot_id=shot_id,
        total=max(0.0, min(1.0, total)),
        embedding=embedding_sim,
        caption=caption_sim,
        character=character_score,
        quality=quality_score,
        size=size_score,
        reuse=reuse_score,
        time_order=time_order_score,
        exclusions=exclusions_score,
    )


def score_candidates(
    shot_ids: list[str],
    shot_entries: dict[str, ShotIndexEntry],
    shot_cast: dict[str, ShotCast],
    shot_captions: dict[str, ShotCaption],
    segment_text: str,
    embedding_scores: dict[str, float],
    character_ids: set[str],
    used_shot_ids: set[str],
    segment_start_ms: int,
    weights: ScoringWeights | None = None,
    size_preference: Literal["long", "medium", "close"] | None = None,
    caption_similarity: Callable[[str, str], float] | None = None,
) -> list[ShotScore]:
    """Score a list of candidate shots, returning sorted by total score (descending).

    Args:
        shot_ids: candidates to score
        shot_entries: shot_id -> ShotIndexEntry (metadata)
        shot_cast: shot_id -> ShotCast (who's in it)
        shot_captions: shot_id -> ShotCaption (descriptions)
        segment_text: narration text to match against
        embedding_scores: shot_id -> vector similarity
        character_ids: character IDs mentioned in text
        used_shot_ids: shots already used
        segment_start_ms: segment start time
        weights: scoring weights
        size_preference: preferred shot size
        caption_similarity: optional similarity function

    Returns:
        List of ShotScore, sorted by total (highest first)
    """
    scores: list[ShotScore] = []
    for shot_id in shot_ids:
        entry = shot_entries.get(shot_id)
        if not entry:
            continue  # skip if entry missing
        score = score_shot(
            shot_id=shot_id,
            shot_entry=entry,
            shot_cast=shot_cast.get(shot_id),
            shot_caption=shot_captions.get(shot_id),
            segment_text=segment_text,
            embedding_scores=embedding_scores,
            character_ids=character_ids,
            used_shot_ids=used_shot_ids,
            segment_start_ms=segment_start_ms,
            weights=weights,
            size_preference=size_preference,
            caption_similarity=caption_similarity,
        )
        scores.append(score)

    # Sort by total descending, then by shot_id for determinism
    scores.sort(key=lambda s: (-s.total, s.shot_id))
    return scores
