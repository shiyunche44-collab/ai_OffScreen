"""LLM-based shot reranking: refine automatic scoring with semantic reasoning.

Given scored candidates and their descriptions, ask LLM to pick the best N shots
in the best order, with explanations. This adds semantic coherence beyond
simple scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from offscreen.algo.scoring import ShotScore
from offscreen.domain.index import ShotCaption


@dataclass(frozen=True)
class ShotChoice:
    """LLM's choice for one position in the sequence."""

    rank: int  # 1, 2, 3, ...
    shot_id: str
    reason: str  # why this shot was chosen


@dataclass(frozen=True)
class RerankResult:
    """Result of LLM-based reranking."""

    choices: list[ShotChoice]  # ordered by rank
    summary: str  # overall reasoning


def build_rerank_prompt(
    segment_text: str,
    scored_shots: list[ShotScore],
    shot_captions: dict[str, ShotCaption],
    target_count: int = 5,
) -> str:
    """Build a prompt for LLM-based shot reranking.

    Args:
        segment_text: narration text for this segment
        scored_shots: list of ShotScore (already ordered by score)
        shot_captions: shot_id -> ShotCaption (descriptions)
        target_count: how many shots to recommend

    Returns:
        Prompt string for LLM
    """
    top_shots = scored_shots[:10]  # consider top 10 for efficiency

    shot_descriptions = []
    for i, score in enumerate(top_shots, 1):
        caption = shot_captions.get(score.shot_id)
        description = caption.caption if caption else "No description"
        shot_descriptions.append(
            f"{i}. {score.shot_id} (score: {score.total:.2f})\n   {description}"
        )

    prompt = f"""Given the following narration segment and candidate shots, \
select and rank the {target_count} best shots that visually match the text.

**Narration:**
{segment_text}

**Candidate shots (ranked by automatic score):**
{chr(10).join(shot_descriptions)}

**Task:**
1. Select up to {target_count} shots that best match the narration
2. Rank them in visual narrative order (how they would flow in the edit)
3. For each shot, briefly explain why it matches the text

**Output format (JSON array):**
[
  {{"shot_id": "sh_XXXX", "reason": "brief reason"}},
  ...
]

Respond with ONLY the JSON array, no other text."""

    return prompt
