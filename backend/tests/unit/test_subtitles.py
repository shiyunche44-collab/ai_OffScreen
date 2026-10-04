from __future__ import annotations

import pytest

from offscreen.algo.subtitles import (
    Cue,
    SubtitleParseError,
    guess_language,
    parse_ass,
    parse_srt,
    parse_subtitles,
)

SRT = """﻿1
00:00:01,000 --> 00:00:03,500
<i>What are you</i>
doing here?

2
00:01:02,250 --> 00:01:04,000
{\\an8}Second cue

3
00:01:05,000 --> 00:01:05,000
zero length, dropped

4
00:02:00.5 --> 00:02:01.75
dot separator
"""

ASS = """[Script Info]
Title: x

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:05.00,0:00:07.50,Default,,0,0,0,,{\\i1}Hello{\\i0}\\Nthere, friend
Dialogue: 0,0:00:01.25,0:00:02.00,Default,,0,0,0,,Earlier, with a comma
Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{\\p1}m 0 0 l 10 0 10 10{\\p0}
Dialogue: 0,0:00:08.00,0:00:08.00,Default,,0,0,0,,empty span
Comment: 0,0:00:09.00,0:00:10.00,Default,,0,0,0,,not dialogue
"""


def test_srt_parses_times_strips_markup_and_skips_empty_spans() -> None:
    assert parse_srt(SRT) == [
        Cue(1000, 3500, "What are you doing here?"),
        Cue(62250, 64000, "Second cue"),
        Cue(120500, 121750, "dot separator"),
    ]


def test_srt_with_windows_line_endings() -> None:
    assert parse_srt("1\r\n00:00:00,000 --> 00:00:01,000\r\nHi\r\n\r\n") == [Cue(0, 1000, "Hi")]


def test_ass_parses_dialogue_sorts_and_drops_drawings() -> None:
    assert parse_ass(ASS) == [
        Cue(1250, 2000, "Earlier, with a comma"),
        Cue(5000, 7500, "Hello there, friend"),
    ]


def test_ass_requires_a_format_line() -> None:
    with pytest.raises(SubtitleParseError, match="Format"):
        parse_ass("[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,x,,0,0,0,,hi\n")


def test_dispatch_by_extension() -> None:
    assert parse_subtitles(SRT, ".SRT")[0].text.startswith("What")
    assert len(parse_subtitles(ASS, ".ssa")) == 2
    with pytest.raises(SubtitleParseError, match="unsupported"):
        parse_subtitles("", ".vtt")


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("你好，今天天气不错", "zh"),
        ("こんにちは、元気ですか", "ja"),
        ("안녕하세요 반갑습니다", "ko"),
        ("What are you doing here?", "und"),
        ("123 ...", "und"),
    ],
)
def test_language_guess(text: str, lang: str) -> None:
    assert guess_language([Cue(0, 1, text)]) == lang
