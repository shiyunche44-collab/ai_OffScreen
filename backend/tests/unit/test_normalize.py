from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.normalize import int_to_cn, normalize_for_speech, remap_timings
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.services.tts import NormalizingTTS


@pytest.mark.parametrize(
    ("n", "cn"),
    [
        (0, "零"), (7, "七"), (10, "十"), (11, "十一"), (20, "二十"), (100, "一百"),
        (101, "一百零一"), (110, "一百一十"), (1000, "一千"), (1024, "一千零二十四"),
        (10000, "一万"), (10001, "一万零一"), (100000, "十万"), (120034, "十二万零三十四"),
        (100000000, "一亿"), (123456789, "一亿二千三百四十五万六千七百八十九"),
    ],
)  # fmt: skip
def test_int_to_cn(n: int, cn: str) -> None:
    assert int_to_cn(n) == cn


@pytest.mark.parametrize(
    ("src", "spoken"),
    [
        ("他活了36岁", "他活了三十六岁"),
        ("2024年夏天", "二零二四年夏天"),
        ("花了3.5万元", "花了三点五万元"),
        ("增长50%", "增长百分之五十"),
        ("温度-5℃", "温度负五摄氏度"),
        ("重3.5kg", "重三点五千克"),
        ("时速120km/h", "时速一百二十千米每小时"),
        ("身高180cm", "身高一百八十厘米"),
        ("1/2的人", "二分之一的人"),
        ("晚上10:30见", "晚上十点三十分见"),
        ("下午3:05", "下午三点零五分"),
        ("8:00开始", "八点开始"),
        ("3-5天", "三到五天"),
        ("2个人", "两个人"),
        ("第2集", "第二集"),
        ("12,345人", "一万二千三百四十五人"),
        ("电话13800138000", "电话一三八零零一三八零零零"),
        ("007特工", "零零七特工"),
        ("A&B", "A和B"),
        ("1+1=2", "一加一等于二"),
        ("**重点**", "重点"),
        ("好😀的", "好的"),
        ("等等.....", "等等…"),
        ("iPhone15", "iPhone15"),
        ("5G网络", "五G网络"),
        ("", ""),
    ],
)  # fmt: skip
def test_normalize_common_cases(src: str, spoken: str) -> None:
    assert normalize_for_speech(src).text == spoken


def test_plain_text_is_untouched() -> None:
    s = "他推开门，看见了她。Hello world!"
    n = normalize_for_speech(s)
    assert n.text == s and n.origin == tuple((i, i + 1) for i in range(len(s)))


@given(st.text(alphabet="0123456789.,%:/-~ 年个kgm℃&+*你好abc", max_size=40))
def test_origin_map_is_valid(src: str) -> None:
    n = normalize_for_speech(src)
    assert len(n.origin) == len(n.text)
    assert all(0 <= lo <= hi <= len(src) for lo, hi in n.origin)
    assert [lo for lo, _ in n.origin] == sorted(lo for lo, _ in n.origin)


@given(st.text(alphabet="0123456789.,%:/-~ 年个kgm℃&+*你好abc", max_size=40))
def test_remapped_timings_one_per_original_char_and_monotonic(src: str) -> None:
    n = normalize_for_speech(src)
    timings = [(i * 100, (i + 1) * 100) for i in range(len(n.text))]
    out = remap_timings(n, timings, len(src), len(n.text) * 100)
    assert len(out) == len(src) if n.text else out == [] or len(out) == len(src)
    prev = 0
    for s, e in out:
        assert prev <= s <= e <= len(n.text) * 100
        prev = e


def test_normalizing_tts_maps_timings_to_original_text(tmp_path: Path) -> None:
    fake = FakeTTS(chars_per_s=5.0)
    tts = NormalizingTTS(fake)
    text = "重3kg"
    out = tts.synthesize(text, voice_id="v")
    assert fake.calls[0][0] == "重三千克"  # the engine only sees the spoken form
    assert len(out.char_timings) == len(text)
    assert out.duration_ms == 800
    assert out.char_timings[0] == (0, 200)  # 重
    assert out.char_timings[-1][1] == 800  # the unit's end lands on the last original char
    again = tts.synthesize("重三千克", voice_id="v")
    assert len(again.char_timings) == 4
