from __future__ import annotations

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.align import MIN_COVERAGE, align_chars, recognized_chars, spoken


def per_char(text: str, ms: int = 100) -> list[tuple[str, int, int]]:
    """Every spoken character recognized as its own word, `ms` each."""
    chars = [c for c in text if spoken(c)]
    return [(c, k * ms, (k + 1) * ms) for k, c in enumerate(chars)]


def test_a_perfect_transcription_gives_its_own_times() -> None:
    text = "他回来了，Hi。"
    a = align_chars(text, per_char(text), 1000)
    assert a.coverage == 1.0 and a.trusted
    assert a.char_timings == [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 400),  # ，
        (400, 500),
        (500, 600),
        (600, 600),  # 。
    ]


def test_words_are_shared_by_their_characters_and_numerals_read_out() -> None:
    assert recognized_chars([("1990年", 0, 500), ("，", 500, 600), ("OK", 600, 800)]) == [
        ("一", 0, 100),
        ("九", 100, 200),
        ("九", 200, 300),
        ("零", 300, 400),
        ("年", 400, 500),
        ("o", 600, 700),
        ("k", 700, 800),
    ]


def test_a_misheard_character_takes_the_time_between_its_neighbours() -> None:
    text = "他出生了"
    a = align_chars(text, [("他出", 0, 400), ("声", 400, 600), ("了", 900, 1000)], 1200)
    assert a.coverage == 0.75
    assert a.char_timings == [(0, 200), (200, 400), (400, 900), (900, 1000)]


def test_unmatched_ends_take_the_usual_time_per_character() -> None:
    text = "啊他来了哦"
    a = align_chars(text, [("他来了", 600, 900)], 2000)  # 100 ms per character
    assert a.char_timings[0] == (500, 600)
    assert a.char_timings[1:4] == [(600, 700), (700, 800), (800, 900)]
    assert a.char_timings[4] == (900, 1000)


def test_little_or_nothing_recognized_is_not_trusted() -> None:
    assert align_chars("他回来了", [], 1000).char_timings == []
    assert align_chars("他回来了", [("，", 0, 10)], 1000).char_timings == []
    poor = align_chars("他回来了吗", [("他", 0, 100), ("你好", 100, 300)], 1000)
    assert poor.coverage < MIN_COVERAGE and not poor.trusted
    assert align_chars("。。", [("他", 0, 100)], 1000).char_timings == []


def test_times_stay_inside_the_audio() -> None:
    a = align_chars("他来了", [("他", 0, 100), ("来了", 100, 5000)], 1200)
    assert a.char_timings[-1][1] == 1200


_CHARS = st.sampled_from(list("他来了走吧小镇一二三，。 AB"))


@given(
    st.text(_CHARS, max_size=25),
    st.lists(
        st.tuples(
            st.text(_CHARS, min_size=1, max_size=4), st.integers(0, 4000), st.integers(0, 800)
        ),
        max_size=12,
    ),
    st.integers(1, 6000),
)
def test_timings_are_whole_monotonic_and_inside_the_audio(
    text: str, raw: list[tuple[str, int, int]], duration: int
) -> None:
    words = sorted((w, s, s + d) for w, s, d in raw)
    a = align_chars(text, words, duration)
    assert 0.0 <= a.coverage <= 1.0
    if a.char_timings:
        assert len(a.char_timings) == len(text)
        assert all(0 <= s <= e <= duration for s, e in a.char_timings)
        assert all(x[1] <= y[0] for x, y in pairwise(a.char_timings))


@given(st.text(_CHARS, min_size=1, max_size=25))
def test_hearing_exactly_the_text_matches_everything(text: str) -> None:
    words = per_char(text)
    a = align_chars(text, words, 100 * len(words) + 1)
    if words:
        assert a.coverage == 1.0
