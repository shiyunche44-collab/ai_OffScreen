from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.textnorm import cardinal, normalize


@pytest.mark.parametrize(
    ("n", "spoken"),
    [
        (0, "零"),
        (7, "七"),
        (10, "十"),
        (12, "十二"),
        (20, "二十"),
        (101, "一百零一"),
        (110, "一百一十"),
        (1001, "一千零一"),
        (1010, "一千零一十"),
        (10010, "一万零一十"),
        (10050, "一万零五十"),
        (11000, "一万一千"),
        (100000, "十万"),
        (110000, "十一万"),
        (1000001, "一百万零一"),
        (100001000, "一亿零一千"),
        (100000010, "一亿零一十"),
    ],
)
def test_cardinal(n: int, spoken: str) -> None:
    assert cardinal(n) == spoken


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        # numbers by value, 两, decimals, negatives, thousands commas
        ("一共1024人", "一共一千零二十四人"),
        ("1,000,000", "一百万"),
        ("2个人", "两个人"),
        ("2 个", "两 个"),
        ("第2个", "第二个"),
        ("12个", "十二个"),
        ("2000人", "两千人"),
        ("20000元", "两万元"),
        ("2，", "二，"),
        ("3.14", "三点一四"),
        ("0.5秒", "零点五秒"),
        ("2.5万", "二点五万"),
        ("温度-5度", "温度负五度"),
        # digit by digit
        ("1990年", "一九九零年"),
        ("1949年10月1日", "一九四九年十月一日"),
        ("B612号", "B六一二号"),
        ("MP3", "MP三"),
        ("代号007", "代号零零七"),
        ("13812345678", "一三八一二三四五六七八"),
        # percent, fractions, dates, clock, ratio, range, score, arithmetic
        ("涨了50%", "涨了百分之五十"),
        ("-5%", "负百分之五"),
        ("3/4的人", "四分之三的人"),
        ("1/0", "一/零"),
        ("2024-13-01", "两千零二十四-十三-零一"),
        ("2024-10-13", "二零二四年十月十三日"),
        ("2024/1/5", "二零二四年一月五日"),
        ("10:05", "十点零五分"),
        ("2:00", "两点"),
        ("23:59:08", "二十三点五十九分零八秒"),
        ("2:1", "二比一"),
        ("99:10", "九十九比十"),
        ("3-5天", "三到五天"),
        ("2~3个", "两到三个"),
        ("3-5km", "三到五公里"),
        ("30%-50%", "百分之三十到百分之五十"),
        ("1-2万", "一万到两万"),
        ("$1-2万", "一万到两万美元"),
        ("1990-1995年", "一九九零到一九九五年"),
        ("比分3-2", "比分三比二"),
        ("1+1=2", "一加一等于二"),
        ("3 × 4", "三乘四"),
        # currency and units
        ("$100", "一百美元"),
        ("¥1.5亿", "一点五亿元"),
        ("€20", "二十欧元"),
        ("100 km", "一百公里"),
        ("120km/h", "一百二十公里每小时"),
        ("5kg", "五公斤"),
        ("180cm", "一百八十厘米"),
        ("36.5℃", "三十六点五摄氏度"),
        ("-40°C", "负四十摄氏度"),
        ("90°", "九十度"),
        ("50m²", "五十平方米"),
        ("5mins", "五mins"),
        # fullwidth input
        ("１２３个", "一百二十三个"),
        ("５０％", "百分之五十"),
        # words and symbols
        ("A & B", "A和B"),
        ("汤姆&杰瑞", "汤姆和杰瑞"),
        ("Mr. Smith", "Mister Smith"),
        ("Dr. Who", "Doctor Who"),
        ("人类vs外星人", "人类对外星人"),
        ("A VS B", "A 对 B"),
        # left alone
        ("Apollo 13号", "Apollo 十三号"),
        ("4K画质", "四K画质"),
        ("FBI探员", "FBI探员"),
        ("这座小镇——没有人知道。", "这座小镇——没有人知道。"),
        ("versus", "versus"),
        ("", ""),
    ],
)
def test_normalize(text: str, spoken: str) -> None:
    assert normalize(text).text == spoken


def test_pieces_cover_both_texts_in_order() -> None:
    s = normalize("在1990年，2个人")
    lo = out = 0
    for p in s.pieces:
        assert (p.src_lo, p.out_lo) == (lo, out)
        assert s.text[p.out_lo : p.out_hi] or p.src_hi > p.src_lo
        lo, out = p.src_hi, p.out_hi
    assert (lo, out) == (len(s.source), len(s.text))


def test_a_replaced_run_shares_the_time_of_what_was_spoken_for_it() -> None:
    s = normalize("约1990年")  # 约 一九九零 年
    spoken = [(k * 100, k * 100 + 100) for k in range(len(s.text))]
    timings = s.source_timings(spoken, 600)
    assert timings[0] == (0, 100)
    assert timings[1:5] == [(100, 200), (200, 300), (300, 400), (400, 500)]
    assert timings[5] == (500, 600)


def test_a_short_source_run_spans_a_long_spoken_one() -> None:
    s = normalize("$5")  # 五美元
    timings = s.source_timings([(0, 100), (100, 200), (200, 300)], 300)
    assert timings == [(0, 150), (150, 300)]


def test_no_engine_timings_means_none() -> None:
    assert normalize("1990年").source_timings([], 1000) == []


def test_timings_must_match_the_spoken_text() -> None:
    with pytest.raises(ValueError):
        normalize("1990年").source_timings([(0, 10)], 100)


_ALPHABET = st.sampled_from(list("0123456789,.:/-~%$¥&+= 年个第字kmgL℃°AB－１％"))


@given(st.text(_ALPHABET, max_size=30), st.integers(1, 5000))
def test_timings_cover_every_source_char_monotonic_within_the_audio(
    text: str, duration_ms: int
) -> None:
    s = normalize(text)
    n = len(s.text)
    spoken = [(duration_ms * k // max(n, 1), duration_ms * (k + 1) // max(n, 1)) for k in range(n)]
    timings = s.source_timings(spoken, duration_ms)
    if not n:
        assert timings == []
        return
    assert len(timings) == len(text)
    prev = 0
    for a, b in timings:
        assert prev <= a <= b <= duration_ms
        prev = b


@given(st.text(_ALPHABET, max_size=30))
def test_normalizing_never_fails_and_is_idempotent_on_digits(text: str) -> None:
    once = normalize(text).text
    assert not any(c in "0123456789" for c in once)
    assert normalize(once).text == once
