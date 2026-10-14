"""Speech text normalization: readings, offset mapping, and the engine wrapper."""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.speech_text import cn_int, normalize_for_speech, remap_timings
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.providers.adapters.speech_normalizer import NormalizingTTS


@pytest.mark.parametrize(
    ("n", "want"),
    [
        (0, "零"), (5, "五"), (10, "十"), (11, "十一"), (20, "二十"), (100, "一百"),
        (105, "一百零五"), (110, "一百一十"), (1205, "一千二百零五"), (1000, "一千"),
        (10000, "一万"), (10005, "一万零五"), (100000, "十万"), (120300, "十二万零三百"),
        (100000000, "一亿"), (123456789, "一亿二千三百四十五万六千七百八十九"),
        (-7, "负七"),
    ],
)  # fmt: skip
def test_cn_int(n: int, want: str) -> None:
    assert cn_int(n) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("他活了83岁", "他活了八十三岁"),
        ("2024年上映", "二零二四年上映"),
        ("3.5倍", "三点五倍"),
        ("增长50%", "增长百分之五十"),
        ("温度-5℃", "温度负五摄氏度"),
        ("3~5天", "三到五天"),
        ("跑了12km", "跑了十二公里"),
        ("时速120km/h", "时速一百二十千米每小时"),
        ("重3kg", "重三千克"),
        ("花了$300", "花了三百美元"),
        ("1,000人", "一千人"),
        ("晚上10:30见", "晚上十点三十分见"),
        ("9:05", "九点零五分"),
        ("3/4的人", "四分之三的人"),
        ("2个人", "两个人"),
        ("12个人", "十二个人"),
        ("房间007", "房间零零七"),
        ("５０人", "五十人"),
        ("她是CEO", "她是C E O"),
        ("来自NASA", "来自NASA"),
        ("AI和人类", "A I和人类"),
        ("hello world", "hello world"),
        ("A&B", "A和B"),
        ("1+1=2", "一加一等于二"),
        ("没有数字。", "没有数字。"),
    ],
)  # fmt: skip
def test_readings(text: str, want: str) -> None:
    assert normalize_for_speech(text).spoken == want


def test_hyphen_between_numbers_is_not_a_minus() -> None:
    assert normalize_for_speech("5-3").spoken == "五-三"


def test_custom_lexicon() -> None:
    assert normalize_for_speech("看GPT", {"GPT": "G P T"}).spoken == "看G P T"
    assert normalize_for_speech("看CEO", {"CEO": "首席执行官"}).spoken == "看首席执行官"


def test_pieces_cover_original() -> None:
    text = "2024年，他跑了12km；AI赢了。"
    nt = normalize_for_speech(text)
    assert nt.pieces[0].lo == 0 and nt.pieces[-1].hi == len(text)
    assert all(a.hi == b.lo for a, b in pairwise(nt.pieces))
    assert "".join(p.spoken for p in nt.pieces) == nt.spoken


@given(st.text(max_size=60))
def test_pieces_always_cover_original(text: str) -> None:
    nt = normalize_for_speech(text)
    assert sum(p.hi - p.lo for p in nt.pieces) == len(text)
    assert "".join(p.spoken for p in nt.pieces) == nt.spoken
    spoken_timings = [(i * 10, i * 10 + 10) for i in range(len(nt.spoken))]
    out = remap_timings(nt.pieces, spoken_timings, len(text))
    if spoken_timings:
        assert len(out) == len(text)
        assert all(a <= b for a, b in out)
        assert all(x[1] <= y[0] or x[1] == y[1] or x[0] <= y[0] for x, y in pairwise(out))


def test_remap_spreads_a_reading_over_its_characters() -> None:
    nt = normalize_for_speech("83岁")  # 八十三岁: 4 spoken chars for 3 original
    timings = [(i * 100, (i + 1) * 100) for i in range(len(nt.spoken))]
    out = remap_timings(nt.pieces, timings, 3)
    assert out[0][0] == 0
    assert out[1][0] == out[0][1] and out[2][0] == out[1][1]  # numeral occupies 0..300
    assert out[1] == (150, 300) or out[0][1] <= 300
    assert out[2] == (300, 400)  # 岁 keeps its own span


def test_remap_without_engine_timings_is_empty() -> None:
    nt = normalize_for_speech("83岁")
    assert remap_timings(nt.pieces, [], 3) == []


def test_wrapper_speaks_normalized_but_times_original() -> None:
    inner = FakeTTS()
    tts = NormalizingTTS(inner)
    audio = tts.synthesize("他活了83岁。", voice_id="v")
    assert inner.calls[0][0] == "他活了八十三岁。"
    assert len(audio.char_timings) == len("他活了83岁。")
    assert tts.id == f"{inner.id}+norm1"


def test_wrapper_passes_plain_text_through() -> None:
    inner = FakeTTS()
    audio = NormalizingTTS(inner).synthesize("你好。", voice_id="v")
    assert inner.calls[0][0] == "你好。"
    assert len(audio.char_timings) == 3
