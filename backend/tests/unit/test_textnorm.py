from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.textnorm import read_integer, read_number, to_spoken
from offscreen.providers.adapters.fake import FakeTTS


@pytest.mark.parametrize(
    ("digits", "said"),
    [
        ("0", "零"),
        ("7", "七"),
        ("10", "十"),
        ("12", "十二"),
        ("20", "二十"),
        ("110", "一百一十"),
        ("205", "二百零五"),
        ("1001", "一千零一"),
        ("1500", "一千五百"),
        ("2024", "两千零二十四"),
        ("10086", "一万零八十六"),
        ("20000", "两万"),
        ("100000", "十万"),
        ("1000000", "一百万"),
        ("12345678", "一千二百三十四万五千六百七十八"),
        ("100000001", "一亿零一"),
        ("300000000", "三亿"),
        ("007", "零零七"),
    ],
)
def test_integers_are_read_as_quantities(digits: str, said: str) -> None:
    assert read_integer(digits) == said


def test_numbers_as_written() -> None:
    assert read_number("3.14") == "三点一四"
    assert read_number("0.5") == "零点五"
    assert read_number("1,234,567") == "一百二十三万四千五百六十七"


@pytest.mark.parametrize(
    ("text", "said"),
    [
        # years and dates
        ("2008年，他37岁。", "二零零八年，他三十七岁。"),
        ("1990-2000年", "一九九零到二零零零年"),
        ("2023-2024赛季", "二零二三到二零二四赛季"),
        ("1990s", "一九九零年代"),
        ("5月3日", "五月三日"),
        # quantities
        ("1,000人", "一千人"),
        ("2个人", "两个人"),
        ("第2次", "第二次"),
        ("2月", "二月"),
        ("2万", "两万"),
        ("3-5个", "三到五个"),
        ("1000-2000人", "一千到两千人"),
        ("3/4的人", "四分之三的人"),
        ("温度-5℃", "温度负五摄氏度"),
        ("#1", "第一"),
        # percentages, money
        ("增长了50%", "增长了百分之五十"),
        ("１２３％", "百分之一百二十三"),
        ("10%-20%", "百分之十到百分之二十"),
        ("票房$5亿", "票房五亿美元"),
        ("¥100", "一百元"),
        # units
        ("走了2.5km", "走了二点五公里"),
        ("长2米", "长两米"),
        ("用了5min", "用了五分钟"),
        ("时速120km/h", "时速一百二十公里每小时"),
        ("37°C", "三十七摄氏度"),
        # clock and scores
        ("10:30出发", "十点三十分出发"),
        ("2:05", "两点零五分"),
        ("9:00", "九点"),
        ("比分3:2", "比分三比二"),
        # Latin letters and symbols
        ("FBI与NASA", "F B I与NASA"),
        ("R2-D2和C-3PO", "R二D二和C三PO"),
        ("T-800", "T八零零"),
        ("4K", "四K"),
        ("Hello world", "Hello world"),
        ("1+1=2", "一加一等于二"),
        ("Tom & Jerry", "Tom和Jerry"),
    ],
)
def test_readings(text: str, said: str) -> None:
    assert to_spoken(text).text == said


def test_plain_narration_is_its_own_spoken_form() -> None:
    text = "这座小镇已经很多年没有下过雪了。可是那天夜里，人们醒来的时候……"
    spoken = to_spoken(text)
    assert spoken.text == text
    timings = [(i * 100, i * 100 + 100) for i in range(len(text))]
    assert spoken.source_timings(timings) == timings


def test_timings_map_back_to_the_source_characters() -> None:
    spoken = to_spoken("他37岁")  # 他 三十七 岁
    assert spoken.text == "他三十七岁"
    timings = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
    # 他 | 3 7 share 三十七 (100-400) | 岁
    assert spoken.source_timings(timings) == [(0, 100), (100, 250), (250, 400), (400, 500)]


def test_a_dropped_character_gets_an_empty_interval() -> None:
    spoken = to_spoken("R2-D2")
    assert spoken.text == "R二D二"
    out = spoken.source_timings([(0, 100), (100, 200), (200, 300), (300, 400)])
    assert len(out) == 5
    assert out[0][0] == 0 and out[-1][1] == 400


def test_no_timestamps_stay_none() -> None:
    assert to_spoken("2008年").source_timings([]) == []


def test_timings_must_match_the_spoken_text() -> None:
    with pytest.raises(ValueError, match="spoken text"):
        to_spoken("37岁").source_timings([(0, 1)] * 3)  # the source length, not 三十七岁


_TEXT = st.text(
    alphabet=st.sampled_from(
        list("他的我们走了年月个第岁，。、！ 0123456789.,:-/%$¥℃°+=&#ABCDFIRTkmgshAaZz２％")
    ),
    max_size=40,
)


@given(_TEXT)
def test_pieces_cover_both_texts(text: str) -> None:
    spoken = to_spoken(text)
    assert "".join(text[p.src_lo : p.src_hi] for p in spoken.pieces) == text
    assert "".join(spoken.text[p.out_lo : p.out_hi] for p in spoken.pieces) == spoken.text
    for a, b in pairwise(spoken.pieces):
        assert a.src_hi == b.src_lo and a.out_hi == b.out_lo


@given(_TEXT.filter(lambda t: any(not c.isspace() for c in t)))
def test_engine_timings_become_one_monotonic_interval_per_source_char(text: str) -> None:
    spoken = to_spoken(text)
    if not spoken.text.strip():
        return
    audio = FakeTTS().synthesize(spoken.text, voice_id="v")
    out = spoken.source_timings(audio.char_timings)
    assert len(out) == len(text)
    assert all(s <= e for s, e in out)
    assert all(a[1] <= b[0] for a, b in pairwise(out))
    assert out[-1][1] <= audio.duration_ms
