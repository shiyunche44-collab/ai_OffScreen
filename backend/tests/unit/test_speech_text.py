from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.speech_text import cardinal, to_spoken


@pytest.mark.parametrize(
    ("n", "want"),
    [
        (0, "零"), (5, "五"), (10, "十"), (12, "十二"), (20, "二十"), (100, "一百"),
        (105, "一百零五"), (110, "一百一十"), (1000, "一千"), (1001, "一千零一"),
        (10000, "一万"), (10001, "一万零一"), (100000, "十万"), (120000, "十二万"),
        (100000000, "一亿"), (100020003, "一亿零二万零三"),
    ],
)  # fmt: skip
def test_cardinal(n: int, want: str) -> None:
    assert cardinal(n) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("50%增长", "百分之五十增长"),
        ("0.5‰", "千分之零点五"),
        ("3.14", "三点一四"),
        ("1,234人", "一千二百三十四人"),
        ("2024年", "二零二四年"),
        ("2024 年", "二零二四 年"),
        ("2024人", "两千零二十四人"),
        ("2个人", "两个人"),
        ("200人", "两百人"),
        ("12000元", "一万两千元"),
        ("22", "二十二"),
        ("007", "零零七"),
        ("13812345678", "一三八一二三四五六七八"),
        ("v1.2.3", "v一点二点三"),
        ("-5℃", "零下五摄氏度"),
        ("25℃", "二十五摄氏度"),
        ("-3", "负三"),
        ("气温-3度", "气温负三度"),
        ("3-5", "三-五"),
        ("$5万", "五万美元"),
        ("¥3.5", "三点五元"),
        ("120km/h", "一百二十公里每小时"),
        ("80 kg", "八十公斤"),
        ("5m", "五米"),
        ("100m²", "一百平方米"),
        ("12:05开会", "十二点零五分开会"),
        ("2:30", "两点三十分"),
        ("12:00", "十二点"),
        ("1/2", "二分之一"),
        ("3~5天", "三到五天"),
        ("１２３", "一百二十三"),
        ("A&B", "A和B"),
        ("**重点**", "重点"),
        ("1990s", "一千九百九十s"),
        ("我有3 apples", "我有三 apples"),
        ("没有数字，也没有英文。", "没有数字，也没有英文。"),
        ("AI时代", "AI时代"),
    ],
)  # fmt: skip
def test_to_spoken(text: str, want: str) -> None:
    assert to_spoken(text).text == want


def test_lexicon_wins_and_respects_word_boundaries() -> None:
    lex = {"iPhone": "爱疯", "重": "chong"}
    assert to_spoken("买iPhone, 不是iPhones", lex).text == "买爱疯, 不是iPhones"
    assert to_spoken("重庆", lex).text == "chong庆"


def test_remap_shares_time_of_a_replaced_span() -> None:
    sp = to_spoken("共50%")
    assert sp.text == "共百分之五十"
    timings = [(i * 100, i * 100 + 100) for i in range(len(sp.text))]
    got = sp.remap_timings(timings)
    assert len(got) == 4
    assert got[0] == (0, 100)
    assert got[1][0] == 100 and got[-1][1] == 600


def test_remap_drops_markup_and_checks_length() -> None:
    sp = to_spoken("**好**")
    assert sp.text == "好"
    got = sp.remap_timings([(0, 50)])
    assert len(got) == 5 and got[2] == (0, 50)
    assert to_spoken("好").remap_timings([]) == []
    with pytest.raises(ValueError):
        sp.remap_timings([(0, 1), (1, 2)])


@given(st.text(max_size=60))
def test_remap_covers_every_source_char_and_is_monotonic(text: str) -> None:
    sp = to_spoken(text.translate({ord(c): None for c in "  "}))
    timings = [(i * 10, i * 10 + 10) for i in range(len(sp.text))]
    got = sp.remap_timings(timings)
    if not timings:
        assert got == []
        return
    assert len(got) == sp.source_len
    prev = 0
    for s, e in got:
        assert prev <= s <= e
        prev = e
