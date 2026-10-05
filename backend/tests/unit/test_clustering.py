"""Grouping face features into people."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from offscreen.algo.clustering import (
    NOISE,
    centroids,
    cluster_faces,
    merge_clusters,
    micro_clusters,
    shot_cast,
    shot_indexes,
)
from offscreen.domain.index import FaceBox, ShotFaces

DIM = 64


def people(count: int, seed: int = 0) -> np.ndarray:
    """`count` unrelated unit directions (distinct people)."""
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(count, DIM))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def faces_of(
    centres: np.ndarray, counts: list[int], noise: float = 0.15, seed: int = 1
) -> tuple[np.ndarray, list[int], list[int]]:
    """Noisy copies of each centre, interleaved in time; returns (vectors, true person, shot)."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[np.ndarray, int, int]] = []
    for person, n in enumerate(counts):
        for k in range(n):
            v = centres[person] + noise * rng.normal(size=DIM) / np.sqrt(DIM) * 3
            rows.append((v / np.linalg.norm(v), person, k))
    rng.shuffle(rows)  # type: ignore[arg-type]
    return (
        np.stack([r[0] for r in rows]),
        [r[1] for r in rows],
        [r[2] for r in rows],  # the k-th sighting of a person is in shot k
    )


def same_partition(a: list[int], b: list[int]) -> bool:
    """Both label lists group the rows identically (label names aside)."""
    pairs = set(zip(a, b, strict=True))
    return len({x for x, _ in pairs}) == len(pairs) == len({y for _, y in pairs})


def test_distinct_people_are_found_and_ordered_by_how_often_they_appear() -> None:
    vectors, truth, shot = faces_of(people(4), [40, 25, 12, 6])
    labels = cluster_faces(vectors, shot)
    assert same_partition(list(labels), truth)
    sizes = [int((labels == c).sum()) for c in range(4)]
    assert sizes == [40, 25, 12, 6]  # label 0 is the most frequent


def test_one_person_seen_in_different_light_stays_one_cluster() -> None:
    vectors, _, shot = faces_of(people(1), [30], noise=0.25)
    assert set(cluster_faces(vectors, shot)) == {0}


def test_scattered_faces_and_single_shot_groups_are_noise() -> None:
    vectors, truth, shot = faces_of(people(3), [20, 15, 1])
    labels = cluster_faces(vectors, shot)
    lone = truth.index(2)
    assert labels[lone] == NOISE
    assert set(labels) == {NOISE, 0, 1}

    # many faces, but all from one shot (e.g. a poster on the wall): not a character
    labels = cluster_faces(vectors[:15], [0] * 15)
    assert set(labels) == {NOISE}


def test_edge_cases() -> None:
    assert cluster_faces(np.zeros((0, DIM)), []).shape == (0,)
    with pytest.raises(ValueError, match="one shot index per face"):
        cluster_faces(people(2), [0])
    # zero vectors do not crash
    assert len(cluster_faces(np.zeros((5, DIM)), [0, 1, 2, 3, 4])) == 5


def test_micro_clusters_split_distant_faces_and_merge_joins_close_ones() -> None:
    a, b = people(2)
    near = a + 0.05 * b
    vectors = np.stack([a, near / np.linalg.norm(near), b])
    labels = micro_clusters(vectors, threshold=0.9)
    assert labels[0] == labels[1] != labels[2]

    split = np.array([0, 1, 2])
    joined = merge_clusters(vectors, split, threshold=0.9)
    assert joined[0] == joined[1] != joined[2]
    assert list(merge_clusters(vectors, split, threshold=1.5)) == [0, 1, 2]  # nothing reaches it


def test_centroids_are_unit_means_per_label() -> None:
    a, b = people(2)
    vectors = np.stack([a, a, b])
    c = centroids(vectors, np.array([0, 0, 1]))
    np.testing.assert_allclose(c, [a, b], atol=1e-9)
    assert centroids(vectors, np.array([NOISE] * 3)).shape == (0, DIM)


@settings(max_examples=30, deadline=None)
@given(st.integers(2, 5), st.integers(0, 50))
def test_clustering_is_deterministic_and_labels_are_dense(k: int, seed: int) -> None:
    vectors, _, shot = faces_of(people(k, seed), [8] * k, seed=seed)
    first = cluster_faces(vectors, shot)
    assert list(first) == list(cluster_faces(vectors, shot))
    found = sorted(set(first) - {NOISE})
    assert found == list(range(len(found)))
    sizes = [int((first == c).sum()) for c in found]
    assert sizes == sorted(sizes, reverse=True)


def box(cid: str | None, area: float) -> FaceBox:
    return FaceBox(character_id=cid, bbox=(0.0, 0.0, 0.1, 0.1), area_ratio=area)


def test_shot_cast_ranks_by_share_of_identified_face_area() -> None:
    cast = shot_cast([box("ch_02", 0.1), box("ch_01", 0.2), box("ch_02", 0.1), box(None, 0.9)])
    assert [c for c, _, _ in cast] == ["ch_01", "ch_02"] or [c for c, _, _ in cast] == [
        "ch_02",
        "ch_01",
    ]
    assert sum(share for _, share, _ in cast) == pytest.approx(1.0)
    by = {c: (share, largest) for c, share, largest in cast}
    assert by["ch_01"] == (pytest.approx(0.5), 0.2) and by["ch_02"] == (pytest.approx(0.5), 0.1)
    assert shot_cast([box(None, 0.3)]) == [] and shot_cast([]) == []
    # a clear leader comes first
    lead = shot_cast([box("ch_03", 0.3), box("ch_01", 0.1)])
    assert [c for c, _, _ in lead] == ["ch_03", "ch_01"]


def test_shot_indexes_follow_document_order() -> None:
    shots = [
        ShotFaces(shot_id="sh_a", faces=[box(None, 0.1)] * 2),
        ShotFaces(shot_id="sh_b", faces=[]),
        ShotFaces(shot_id="sh_c", faces=[box(None, 0.1)]),
    ]
    assert shot_indexes(shots) == [0, 0, 2]
