from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.textnorm import normalize_for_tts, read_int, read_number


@pytest.mark.parametrize(
    ("n", "reading"),
    [
        (0, "零"),
        (7, "七"),
        (10, "十"),
        (15, "十五"),
        (20, "二十"),
        (105, "一百零五"),
        (110, "一百一十"),
        (1010, "一千零一十"),
        (2000, "两千"),
        (10005, "一万零五"),
        (12000, "一万二千"),
        (100000, "十万"),
        (100010000, "一亿零一万"),
        (-3, "负三"),
    ],
)
def test_whole_numbers(n: int, reading: str) -> None:
    assert read_int(n) == reading


def test_codes_and_long_numbers_are_read_digit_by_digit() -> None:
    assert read_number("007") == "零零七"
    assert read_number("13800138000") == "一三八零零一三八零零零"
    assert read_number("1,234.05") == "一千二百三十四点零五"


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        # numbers
        ("他花了1,200万美元", "他花了一千二百万美元"),
        ("圆周率是3.14", "圆周率是三点一四"),
        ("有2个苹果，2万人", "有两个苹果，两万人"),
        ("第2个人", "第二个人"),
        ("2月", "二月"),
        ("1+1=2", "一加一等于二"),
        # years and dates
        ("2024年", "二零二四年"),
        ("1990-2000年", "一九九零到二零零零年"),
        ("2024-01-05", "二零二四年一月五日"),
        ("80年代", "八十年代"),
        # percentages, fractions, ranges, signs
        ("涨了50%", "涨了百分之五十"),
        ("10-20%", "百分之十到百分之二十"),
        ("1/3的人", "三分之一的人"),
        ("3-5个", "三到五个"),
        ("3~5km", "三到五公里"),
        ("气温-5℃", "气温负五摄氏度"),
        # clock times
        ("10:30见", "十点三十分见"),
        ("9:05到", "九点零五分到"),
        ("8:00", "八点"),
        ("2:15", "两点十五分"),
        # currencies and units
        ("$3亿", "三亿美元"),
        ("￥50", "五十元"),
        ("100km", "一百公里"),
        ("36.5℃", "三十六点五摄氏度"),
        ("120km/h", "一百二十公里每小时"),
        ("跑了100m", "跑了一百米"),
        ("5 min", "五 min"),
        # Latin letters
        ("FBI和NASA", "F B I和NASA"),
        ("A vs B", "A 对 B"),
        ("iPhone很贵", "iPhone很贵"),
        # symbols
        ("Tom & Jerry", "Tom和Jerry"),
        ("#3", "三号"),
        ("#3号", "三号"),
        ("**重点**", "重点"),
        ("好啊~", "好啊"),
        # fullwidth forms
        ("２０２４年", "二零二四年"),
        ("ＦＢＩ", "F B I"),
        # left alone
        ("他走遍了世界。", "他走遍了世界。"),
        ("……", "……"),
    ],
)
def test_readings(text: str, spoken: str) -> None:
    assert normalize_for_tts(text).text == spoken


def test_timings_map_back_to_the_source_characters() -> None:
    norm = normalize_for_tts("花了50%钱")
    assert norm.text == "花了百分之五十钱"
    # 100 ms per spoken character
    timings = [(i * 100, (i + 1) * 100) for i in range(len(norm.text))]
    src = norm.source_timings(timings)
    assert len(src) == len("花了50%钱")
    assert src[0] == (0, 100)
    assert src[1] == (100, 200)
    # "50%" is read in 百分之五十 (200–700 ms), split evenly over its three characters
    assert src[2:5] == [(200, 366), (366, 533), (533, 700)]
    assert src[5] == (700, 800)


def test_dropped_characters_get_an_empty_interval() -> None:
    norm = normalize_for_tts("好*的")
    assert norm.text == "好的"
    assert norm.source_timings([(0, 100), (100, 200)]) == [(0, 100), (100, 100), (100, 200)]


def test_no_timings_stay_no_timings() -> None:
    assert normalize_for_tts("2024年").source_timings([]) == []


def test_timings_must_match_the_spoken_text() -> None:
    with pytest.raises(ValueError):
        normalize_for_tts("2024年").source_timings([(0, 100)])


_TEXT = st.text(
    alphabet=st.sampled_from(list("0123456789,.-+%:/~#*&$￥℃kmAZvs年个第万 你好，。")),
    max_size=40,
)


@given(_TEXT)
def test_pieces_cover_both_texts(text: str) -> None:
    norm = normalize_for_tts(text)
    assert norm.source == text
    src, out = 0, 0
    for p in norm.pieces:
        assert (p.src_lo, p.out_lo) == (src, out)
        assert p.src_hi >= p.src_lo and p.out_hi >= p.out_lo
        src, out = p.src_hi, p.out_hi
    assert (src, out) == (len(text), len(norm.text))


@given(_TEXT, st.integers(min_value=1, max_value=300))
def test_source_timings_are_complete_and_monotonic(text: str, step: int) -> None:
    norm = normalize_for_tts(text)
    timings = [(i * step, (i + 1) * step) for i in range(len(norm.text))]
    src = norm.source_timings(timings)
    if not norm.text:
        assert src == []
        return
    assert len(src) == len(text)
    assert all(a <= b for a, b in src)
    assert all(b1 <= a2 for (_, b1), (a2, _) in pairwise(src))
    assert src[-1][1] <= len(norm.text) * step


@given(_TEXT)
def test_every_digit_gets_a_reading(text: str) -> None:
    """No digit is left for the engine to read in its own way."""
    assert not any(c.isdigit() for c in normalize_for_tts(text).text)
