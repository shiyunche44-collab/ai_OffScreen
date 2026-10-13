"""Text normalization before TTS: how numbers, English, symbols and units are read. Pure.

Narration is written for the eye ("2023年", "50%", "FBI", "3km"); engines guess at such
tokens and often guess wrong. `spoken_text` rewrites them into what should be said
("二零二三年", "百分之五十", "F B I", "三公里") and remembers which source characters every
piece of the spoken text came from, so the engine's per-character timestamps can be mapped
back onto the original text (captions show what was written, not what was said).

Bump `NORMALIZER_VERSION` whenever a reading changes; the spoken text is part of the TTS
cache key, so changed readings resynthesize by themselves."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from offscreen.algo.tts import WordSpan, char_timings_from_spans

NORMALIZER_VERSION = 1

_DIGITS = "零一二三四五六七八九"
_SECTION_UNITS = ("", "万", "亿", "万亿")
_MAX_CARDINAL_DIGITS = 16
"""Longer integers are read digit by digit (they are codes, not quantities)."""

_CLASSIFIERS = frozenset(
    "个只位名天年次部件条本张台辆种项岁口头匹双对场集章页段家间座道颗根把块层级倍首届句步声周"
)
"""A bare 2 before these reads 两 ("2个人" -> 两个人)."""

_UNITS = {
    "km/h": "公里每小时",
    "km²": "平方公里",
    "m²": "平方米",
    "km": "公里",
    "kg": "公斤",
    "cm": "厘米",
    "mm": "毫米",
    "ml": "毫升",
    "mg": "毫克",
    "m": "米",
    "℃": "摄氏度",
    "°C": "摄氏度",
    "°F": "华氏度",
    "°": "度",
}
_CURRENCIES = {"$": "美元", "¥": "元", "￥": "元", "€": "欧元", "£": "英镑"}
_WORDLIKE_ACRONYMS = frozenset({"NASA", "NATO", "UNESCO", "OPEC", "FIFA", "UEFA", "COVID", "ASEAN"})
"""All-caps words said as words; every other 2–5 letter all-caps token is spelled out."""

_SYMBOLS = {"&": "和", "=": "等于", "+": "加"}
_DROPPED = frozenset("*#_|\\^`~～")
"""Markup-ish symbols engines read aloud ("星号") or stumble on; they are not spoken."""

# Full-width digits, letters and a few signs become their ASCII forms. One char to one char,
# so indices into the text do not move.
_HALF_WIDTH = str.maketrans(
    {
        **{chr(0xFF10 + i): str(i) for i in range(10)},
        **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
        **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
        "％": "%",
        "＄": "$",
        "＋": "+",
        "＝": "=",
        "＆": "&",
    }
)

_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_TOKEN = re.compile(
    rf"""
    (?P<time>(?<![\d.:])(?:[01]?\d|2[0-4])[:：][0-5]\d(?![\d:]))
  | (?P<ratio>(?<![\d.:])\d+[:：]\d+(?![\d:]))
  | (?P<frac>(?<![\d./])\d+/\d+(?![\d/]))
  | (?P<range>(?<![\d.])(?:{_NUM})\s*[-~～—–]\s*(?:{_NUM})(?![\d.]))
  | (?P<cur>[$¥￥€£])(?P<curnum>{_NUM})(?P<curmag>[万亿]?)
  | (?P<neg>(?<![0-9A-Za-z.])-)?(?P<num>{_NUM})
      (?:\s?(?P<pct>[%‰])|(?P<unit>{_UNIT_RE})(?![A-Za-z]))?
  | (?P<acro>(?<![A-Za-z])[A-Z]{{2,5}}(?![A-Za-z]))
  | (?P<sym>[&=+])
  | (?P<drop>[{re.escape("".join(sorted(_DROPPED)))}])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Piece:
    """Spoken `text[out_lo:out_hi]` says source `text[src_lo:src_hi]`."""

    src_lo: int
    src_hi: int
    out_lo: int
    out_hi: int


@dataclass(frozen=True)
class SpokenText:
    text: str
    """What the engine is asked to say."""
    source_len: int
    pieces: tuple[Piece, ...]
    """Cover both texts in order, without gaps or overlaps."""

    def source_timings(
        self, timings: Sequence[tuple[int, int]], duration_ms: int
    ) -> list[tuple[int, int]]:
        """Per-character timings of the spoken text -> per-character timings of the source.

        A source token spoken as several characters gets the span of all of them, divided
        evenly among its own characters; a source character that is not spoken gets an empty
        interval where the previous one ended. Empty in, empty out (engine gave no
        timestamps)."""
        if not timings:
            return []
        if len(timings) != len(self.text):
            raise ValueError("timings must have one entry per character of the spoken text")
        spans = [
            WordSpan(timings[p.out_lo][0], timings[p.out_hi - 1][1], p.src_lo, p.src_hi)
            for p in self.pieces
            if p.out_hi > p.out_lo and p.src_hi > p.src_lo
        ]
        return char_timings_from_spans(self.source_len, spans, duration_ms)


def spoken_text(text: str) -> SpokenText:
    """Rewrite `text` into what the narrator should say (see module docstring)."""
    src = text.translate(_HALF_WIDTH)
    out: list[str] = []
    pieces: list[Piece] = []
    pos = 0  # length of the spoken text so far

    def emit(src_lo: int, src_hi: int, said: str) -> None:
        nonlocal pos
        out.append(said)
        pieces.append(Piece(src_lo, src_hi, pos, pos + len(said)))
        pos += len(said)

    done = 0
    for m in _TOKEN.finditer(src):
        for i in range(done, m.start()):
            emit(i, i + 1, src[i])
        said = _read(m, src)
        if len(said) == m.end() - m.start():  # read char for char ("2023" -> 二零二三)
            for k, c in enumerate(said):
                emit(m.start() + k, m.start() + k + 1, c)
        else:
            emit(m.start(), m.end(), said)
        done = m.end()
    for i in range(done, len(src)):
        emit(i, i + 1, src[i])
    return SpokenText(text="".join(out), source_len=len(text), pieces=tuple(pieces))


# ---- readings -----------------------------------------------------------------------------


def _read(m: re.Match[str], src: str) -> str:
    token = m.group(0)
    after = src[m.end() : m.end() + 1]
    if m.group("time"):
        h, mm = re.split("[:：]", token)
        minutes = int(mm)
        if minutes == 0:
            return cardinal(int(h)) + "点"
        return cardinal(int(h)) + "点" + ("零" if minutes < 10 else "") + cardinal(minutes) + "分"
    if m.group("ratio"):
        a, b = re.split("[:：]", token)
        return _number(a, "", "") + "比" + _number(b, "", "")
    if m.group("frac"):
        a, b = token.split("/")
        return _number(b, "", "") + "分之" + _number(a, "", "")
    if m.group("range"):
        a, b = (s.strip() for s in re.split("[-~～—–]", token, maxsplit=1))
        before = src[m.start() - 1 : m.start()]
        return _number(a, before, after) + "到" + _number(b, "", after)
    if m.group("cur"):
        return (
            _number(m.group("curnum"), "", "")
            + (m.group("curmag") or "")
            + _CURRENCIES[m.group("cur")]
        )
    if m.group("num"):
        neg = "负" if m.group("neg") else ""
        before = src[m.start() - 1 : m.start()]
        if m.group("pct"):
            share = "百分之" if m.group("pct") == "%" else "千分之"
            return neg + share + _number(m.group("num"), "", "")
        if m.group("unit"):
            return neg + _number(m.group("num"), before, "") + _UNITS[m.group("unit")]
        return neg + _number(m.group("num"), before, after)
    if m.group("acro"):
        return token if token in _WORDLIKE_ACRONYMS else " ".join(token)
    if m.group("sym"):
        return _SYMBOLS[token]
    return ""  # dropped symbol


def _number(token: str, before: str, after: str) -> str:
    """One written number, read in its context (the characters around it)."""
    token = token.replace(",", "")
    whole, _, frac = token.partition(".")
    if frac:
        return cardinal(int(whole)) + "点" + digits(frac)
    if (
        (len(whole) > 1 and whole.startswith("0"))
        or len(whole) > _MAX_CARDINAL_DIGITS
        or (before.isascii() and before.isalpha())  # model numbers: A320, MP3
    ):
        return digits(whole)
    if len(whole) == 4 and after == "年":
        return digits(whole)
    if whole == "2" and after in _CLASSIFIERS and before != "第":
        return "两"
    return cardinal(int(whole))


def digits(s: str) -> str:
    """Digit by digit: "2023" -> 二零二三."""
    return "".join(_DIGITS[int(c)] for c in s)


def cardinal(n: int) -> str:
    """Chinese cardinal reading of a non-negative integer below 10^16."""
    if n == 0:
        return "零"
    sections: list[int] = []
    while n:
        sections.append(n % 10000)
        n //= 10000
    out = ""
    gap = False  # a zero section or leading zeros since the last thing said
    for i in reversed(range(len(sections))):
        s = sections[i]
        if s == 0:
            gap = gap or bool(out)
            continue
        if out and (gap or s < 1000):
            out += "零"
        out += _section(s) + _SECTION_UNITS[i]
        gap = False
    if out.startswith("一十"):
        out = out[1:]
    if out[:2] in ("二千", "二万", "二亿"):
        out = "两" + out[1:]
    return out


def _section(n: int) -> str:
    out = ""
    zero = False
    for place, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // place % 10
        if d == 0:
            zero = zero or bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        out += _DIGITS[d] + unit
    return out
