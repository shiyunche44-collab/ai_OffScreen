"""Candidate shot generation for segment editing: assemble shots from multiple sources.

For each segment, gather shots from three sources:
1. Scene-referenced shots: all shots in scenes the segment's script text refers to
2. Vector search: text-to-description similarity top-K
3. Character appearance: shots where named characters appear
Then deduplicate and fuse rankings via reciprocal rank fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from offscreen.algo.search import fuse
from offscreen.domain.index import Cast, Characters, ShotCaption, Shots
from offscreen.domain.script import ScriptSegment


@dataclass(frozen=True)
class CandidateSet:
    """Deduplicated candidates with source tracking for debugging."""

    shot_ids: list[str]  # deduplicated, in RRF score order
    sources: dict[str, list[str]]  # shot_id -> list of sources it came from
    scores: dict[str, float]  # shot_id -> RRF score


def candidates_for_segment(
    segment: ScriptSegment,
    shots: Shots,
    scenes_by_id: Mapping[str, Sequence[str]],
    shot_captions: Mapping[str, ShotCaption],
    cast: Cast,
    characters: Characters,
    search_results: Mapping[str, Sequence[str]] | None = None,
) -> CandidateSet:
    """Gather candidates from all sources for one narration/original segment.

    Args:
        segment: The script segment being planned
        shots: All shots in the asset (for duration bounds)
        scenes_by_id: scene_id -> list of shot_ids in that scene
        shot_captions: shot_id -> ShotCaption (descriptions)
        cast: ShotCast data (who appears in each shot)
        characters: Character registry for name lookup
        search_results: search results from the embedding service (shot_id lists by source)

    Returns:
        Deduplicated candidates with RRF fusion score
    """
    rankings: dict[str, list[str]] = {}

    # Source 1: shots from scene_refs
    scene_shots = _shots_from_scenes(segment.scene_refs, scenes_by_id)
    if scene_shots:
        rankings["scenes"] = scene_shots

    # Source 2: vector search (if available)
    if search_results and "embedding" in search_results:
        rankings["embedding"] = list(search_results["embedding"])

    # Source 3: shots where named characters appear
    character_shots = _shots_with_characters(segment.text or "", cast, characters)
    if character_shots:
        rankings["characters"] = character_shots

    # If nothing was gathered, fall back to all shots (in chronological order)
    if not rankings:
        rankings["all"] = [s.id for s in shots.shots]

    # Fuse rankings and deduplicate
    fused = fuse(rankings)
    shot_ids = [shot_id for shot_id, _ in fused]
    scores = {shot_id: score for shot_id, score in fused}

    # Track which sources each shot came from
    sources: dict[str, list[str]] = {}
    for source_name, source_ids in rankings.items():
        for shot_id in source_ids:
            if shot_id not in sources:
                sources[shot_id] = []
            sources[shot_id].append(source_name)

    return CandidateSet(shot_ids=shot_ids, sources=sources, scores=scores)


def _shots_from_scenes(
    scene_refs: list[str], scenes_by_id: Mapping[str, Sequence[str]]
) -> list[str]:
    """Collect all unique shots from referenced scenes, in chronological order."""
    shot_ids: list[str] = []
    seen = set()
    for scene_id in scene_refs:
        for shot_id in scenes_by_id.get(scene_id, []):
            if shot_id not in seen:
                shot_ids.append(shot_id)
                seen.add(shot_id)
    return shot_ids


def _shots_with_characters(text: str, cast: Cast, characters: Characters) -> list[str]:
    """Find shots where characters mentioned in the text appear.

    Extracts character names from text (heuristic: looks in character registry for matches)
    and returns shots ranked by their main cast member's appearance."""
    if not text or not cast.shots:
        return []

    # Build a lookup: character name -> character_id
    name_to_id: dict[str, str] = {}
    for char in characters.characters:
        if char.name:
            name_to_id[char.name.lower()] = char.id
        for alias in char.aliases:
            name_to_id[alias.lower()] = char.id

    # Find character IDs mentioned in the text (simple substring match)
    mentioned: set[str] = set()
    for char_name_lower, char_id in name_to_id.items():
        if char_name_lower in text.lower():
            mentioned.add(char_id)

    if not mentioned:
        return []

    # Collect shots where any mentioned character appears, ranked by share
    shot_scores: list[tuple[str, float]] = []
    for shot_cast in cast.shots:
        for member in shot_cast.characters:
            if member.character_id in mentioned:
                shot_scores.append((shot_cast.shot_id, member.share))
                break  # one shot per character mention (take the lead)

    # Sort by share (descending) then by shot_id for determinism
    shot_scores.sort(key=lambda x: (-x[1], x[0]))
    return [shot_id for shot_id, _ in shot_scores]
