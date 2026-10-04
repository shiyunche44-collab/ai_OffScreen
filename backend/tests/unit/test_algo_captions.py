from __future__ import annotations

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.captions import _spoken, split_caption_lines


def even(text: str, ms_per_char: int = 100) -> list[tuple[int, int]]:
    return [(i * ms_per_char, (i + 1) * ms_per_char) for i in range(len(text))]


def lines(text: str, max_chars: int = 14) -> list[tuple[str, int, int]]:
    return [
        (x.text, x.start_ms, x.end_ms) for x in split_caption_lines(text, even(text), 0, max_chars)
    ]


def test_breaks_after_sentences_and_packs_commas() -> None:
    t = "她走进雪山。小龙倒在地上，奄奄一息！她抱起它。"
    assert lines(t) == [
        ("她走进雪山", 0, 500),
        ("小龙倒在地上，奄奄一息", 600, 1700),
        ("她抱起它", 1800, 2200),
    ]


def test_long_sentence_breaks_at_commas_within_limit() -> None:
    t = "他翻过了一座又一座山，渡过了一条又一条河，终于来到了龙的面前。"
    got = [x[0] for x in lines(t, 14)]
    assert got == ["他翻过了一座又一座山", "渡过了一条又一条河", "终于来到了龙的面前"]


def test_unpunctuated_run_is_cut_into_even_pieces() -> None:
    t = "一二三四五六七八九十甲乙丙丁戊己庚辛壬"  # 19 chars, limit 10 -> 2 pieces
    got = [x[0] for x in lines(t, 10)]
    assert got == ["一二三四五六七八九十", "甲乙丙丁戊己庚辛壬"] or [len(g) for g in got] == [10, 9]
    assert "".join(got) == t


def test_without_timings_chars_share_the_duration() -> None:
    out = split_caption_lines("你好。世界。", [], 1000, 10)
    assert [(x.text, x.start_ms, x.end_ms) for x in out] == [("你好", 0, 500), ("世界", 500, 1000)]


def test_no_spoken_text_gives_no_lines() -> None:
    assert split_caption_lines("……！ ", even("……！ "), 500, 10) == []
    assert split_caption_lines("", [], 500, 10) == []


@given(
    st.text(alphabet="甲乙丙丁戊己庚辛，。！？、 ", min_size=0, max_size=120),
    st.integers(2, 25),
)
def test_line_properties(text: str, limit: int) -> None:
    out = split_caption_lines(text, even(text), 0, limit)
    spoken = "".join(c for c in text if _spoken(c))
    # Every spoken character appears exactly once, in order, and no line is too long.
    assert "".join(c for x in out for c in x.text if _spoken(c)) == spoken
    assert all(1 <= sum(_spoken(c) for c in x.text) <= limit for x in out)
    assert all(x.start_ms <= x.end_ms for x in out)
    assert all(a.end_ms <= b.start_ms for a, b in pairwise(out))
