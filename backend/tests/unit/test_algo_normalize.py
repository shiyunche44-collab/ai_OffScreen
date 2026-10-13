from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.normalize import cardinal, digits, spoken_text
from offscreen.providers.adapters.fake import FakeTTS


@pytest.mark.parametrize(
    ("n", "said"),
    [
        (0, "零"),
        (7, "七"),
        (10, "十"),
        (15, "十五"),
        (20, "二十"),
        (110, "一百一十"),
        (1005, "一千零五"),
        (2000, "两千"),
        (10010, "一万零一十"),
        (100000, "十万"),
        (12345, "一万二千三百四十五"),
        (20000000, "两千万"),
        (100000001, "一亿零一"),
        (1002003000, "十亿零二百万三千"),
    ],
)
def test_cardinal(n: int, said: str) -> None:
    assert cardinal(n) == said


def test_digits() -> None:
    assert digits("2023") == "二零二三"
    assert digits("007") == "零零七"


@pytest.mark.parametrize(
    ("written", "said"),
    [
        # numbers in context
        ("2023年上映", "二零二三年上映"),
        ("90年代", "九十年代"),
        ("等了3年", "等了三年"),
        ("2个人", "两个人"),
        ("第2个", "第二个"),
        ("2000人", "两千人"),
        ("12,345人", "一万二千三百四十五人"),
        ("3.14", "三点一四"),
        ("0.5秒", "零点五秒"),
        ("007号", "零零七号"),
        ("A320客机", "A三二零客机"),
        ("1234567890123456789", "一二三四五六七八九零一二三四五六七八九"),
        # percent, fractions, ratios, ranges, times
        ("50%", "百分之五十"),
        ("3.5%的人", "百分之三点五的人"),
        ("5‰", "千分之五"),
        ("1/3的人", "三分之一的人"),
        ("比分3:2", "比分三比二"),
        ("3-5个", "三到五个"),
        ("2~3天", "两到三天"),
        ("2023-2024年", "二零二三到二零二四年"),
        ("10:30见", "十点三十分见"),
        ("早上8:05", "早上八点零五分"),
        ("23:00", "二十三点"),
        ("零下-5度", "零下负五度"),
        # units and currencies
        ("跑了3km", "跑了三公里"),
        ("100km/h", "一百公里每小时"),
        ("70kg", "七十公斤"),
        ("1.8m高", "一点八米高"),
        ("5min", "五min"),
        ("38℃", "三十八摄氏度"),
        ("30°C", "三十摄氏度"),
        ("$100万", "一百万美元"),
        ("￥50", "五十元"),
        ("500m²", "五百平方米"),
        # English and symbols
        ("FBI探员", "F B I探员"),
        ("AI时代", "A I时代"),
        ("NASA发射", "NASA发射"),
        ("iPhone很贵", "iPhone很贵"),
        ("Tom和Jerry", "Tom和Jerry"),
        ("A&B", "A和B"),
        ("1+1=2", "一加一等于二"),
        ("**重点**", "重点"),
        ("#标签", "标签"),
        # full width
        ("５０％", "百分之五十"),
        ("ＦＢＩ", "F B I"),
        # untouched
        ("他转身离开了。", "他转身离开了。"),
        ("", ""),
    ],
)
def test_spoken_text(written: str, said: str) -> None:
    assert spoken_text(written).text == said


def test_pieces_cover_both_texts() -> None:
    sp = spoken_text("他在2023年花了$5万，约50%。")
    src_at = out_at = 0
    for p in sp.pieces:
        assert (p.src_lo, p.out_lo) == (src_at, out_at)
        src_at, out_at = p.src_hi, p.out_hi
    assert src_at == sp.source_len
    assert out_at == len(sp.text)


def test_source_timings_map_tokens_back() -> None:
    written = "共2023人"
    sp = spoken_text(written)
    assert sp.text == "共两千零二十三人"
    # one 100 ms slot per spoken character
    timings = [(i * 100, (i + 1) * 100) for i in range(len(sp.text))]
    mapped = sp.source_timings(timings, 800)
    assert len(mapped) == len(written)
    assert mapped[0] == (0, 100)  # 共
    # "2023" took 六 spoken chars (100..700), shared evenly by its four digits
    assert mapped[1:5] == [(100, 250), (250, 400), (400, 550), (550, 700)]
    assert mapped[5] == (700, 800)  # 人


def test_unspoken_characters_get_empty_intervals() -> None:
    sp = spoken_text("**好**")
    timings = [(0, 200)]
    assert sp.source_timings(timings, 200) == [(0, 0), (0, 0), (0, 200), (200, 200), (200, 200)]


def test_source_timings_without_engine_timestamps() -> None:
    assert spoken_text("50%").source_timings([], 1000) == []


def test_source_timings_reject_wrong_length() -> None:
    with pytest.raises(ValueError):
        spoken_text("50%").source_timings([(0, 1)], 1000)


_TEXT = st.text(
    alphabet=st.sampled_from(list("他说2023年50%3.5kmFBI-:/,$&*～０１ 好。")), max_size=40
)


@given(_TEXT)
def test_source_timings_are_complete_and_monotonic(text: str) -> None:
    sp = spoken_text(text)
    if not sp.text.strip():
        return
    audio = FakeTTS().synthesize(sp.text, voice_id="v")
    mapped = sp.source_timings(audio.char_timings, audio.duration_ms)
    assert len(mapped) == len(text)
    prev_end = 0
    for start, end in mapped:
        assert prev_end <= start <= end <= audio.duration_ms
        prev_end = end


@given(_TEXT)
def test_normalization_is_idempotent(text: str) -> None:
    once = spoken_text(text).text
    assert spoken_text(once).text == once
