"""Text normalization before synthesis: how digits, Latin letters, symbols and units are read
in Chinese narration (ARCHITECTURE §7.3). Pure.

The engine speaks the normalized text, but the plan keeps the script's text and one timing per
character *of that text* (subtitles are cut from it). So the result also records which part of
the source each part of the spoken text came from, and `Normalized.source_timings` carries the
engine's per-character timings back onto the source characters.

What is rewritten (everything else is passed through unchanged):

- numbers: `1,200` 一千二百, `3.14` 三点一四, `2个` 两个 (a bare 2 before a measure word, not
  after 第), `2000` 两千, `007` 零零七 (a leading zero, or 11+ digits, is read digit by digit)
- years: `2024年` 二零二四年, `1990-2000年` 一九九零到二零零零年
- percentages, fractions, ranges, signs: `50%` 百分之五十, `1/3` 三分之一, `3-5` 三到五,
  `-5` 负五, `1+1=2` 一加一等于二
- clock times: `10:30` 十点三十分, `9:05` 九点零五分, `8:00` 八点
- currencies and units after or before a number: `$3亿` 三亿美元, `100km` 一百公里,
  `36.5℃` 三十六点五摄氏度, `120km/h` 一百二十公里每小时
- Latin letters: all-caps abbreviations are spelled (`FBI` → `F B I`) unless they are commonly
  said as a word (`NASA`), `vs` 对; other words are left for the engine
- symbols: `&` 和, `#3` 三号; markup-like characters with no reading (`*`, `_`, `~` outside a
  range, `#` without a number) are dropped
- fullwidth digits and letters count as their ASCII forms
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

VERSION = 1
"""Bump when the output changes for some input: it is part of what the audio depends on."""

DIGITS = "零一二三四五六七八九"

MEASURE_WORDS = frozenset(
    "个人只次天位名部条本张把辆匹场份件种句岁倍层家双对座周年分秒台集季颗棵枚封首间所"
)
"""After these a bare 2 is read 两 (`2个` 两个), as a count; so it is before 千万亿."""
_LIANG_BEFORE = MEASURE_WORDS | frozenset("千万亿")

CURRENCIES = {"$": "美元", "¥": "元", "￥": "元", "€": "欧元", "£": "英镑"}

UNITS = {
    "km/h": "公里每小时",
    "km": "公里",
    "kg": "公斤",
    "cm": "厘米",
    "mm": "毫米",
    "ml": "毫升",
    "mL": "毫升",
    "m": "米",
    "°C": "摄氏度",
    "℃": "摄氏度",
    "°F": "华氏度",
    "℉": "华氏度",
    "°": "度",
}

SAID_AS_WORDS = frozenset(
    {"NASA", "NATO", "UNESCO", "UNICEF", "OPEC", "AIDS", "SARS", "COVID", "LASER", "RADAR"}
)
"""All-caps abbreviations that are pronounced, not spelled."""

DROPPED = frozenset("*_~～#|")
"""Characters with no reading in narration; dropped when no other rule uses them."""


@dataclass(frozen=True)
class Piece:
    """`text[out_lo:out_hi]` of the normalized text is how `source[src_lo:src_hi]` is read."""

    src_lo: int
    src_hi: int
    out_lo: int
    out_hi: int


@dataclass(frozen=True)
class Normalized:
    source: str
    text: str
    """What the engine is given."""
    pieces: tuple[Piece, ...]
    """In order, covering both `source` and `text` without gaps or overlap."""

    def source_timings(self, timings: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Per-character `(start_ms, end_ms)` of the normalized text (as the engine reports
        them) → the same for each character of the source. A source span read as several
        characters takes the time from the first one's start to the last one's end, divided
        evenly among its characters; a dropped character gets an empty interval where the
        previous one ended. Empty in, empty out (the engine gave no timings)."""
        if not timings:
            return []
        if len(timings) != len(self.text):
            raise ValueError(f"{len(timings)} timings for {len(self.text)} characters")
        out: list[tuple[int, int]] = []
        prev_end = 0
        for p in self.pieces:
            n = p.src_hi - p.src_lo
            if n == 0:
                continue
            if p.out_hi > p.out_lo:
                a = max(timings[p.out_lo][0], prev_end)
                b = max(timings[p.out_hi - 1][1], a)
            else:
                a = b = prev_end
            for k in range(n):
                out.append((a + (b - a) * k // n, a + (b - a) * (k + 1) // n))
            prev_end = b
        return out


# ---- numbers ---------------------------------------------------------------------------


def read_digits(digits: str) -> str:
    """`2024` → 二零二四."""
    return "".join(DIGITS[int(d)] for d in digits)


def _read_section(n: int) -> str:
    """0 < n < 10000, without the leading 一 of 十 dropped (the caller decides)."""
    out = ""
    zero = False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // value % 10
        if d == 0:
            zero = bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        out += DIGITS[d] + unit
    return out


def read_int(n: int) -> str:
    """Chinese reading of a whole number: 10 十, 105 一百零五, 12000 一万二千, 2000 两千."""
    if n < 0:
        return "负" + read_int(-n)
    if n == 0:
        return "零"
    sections: list[int] = []
    while n:
        sections.append(n % 10000)
        n //= 10000
    out = ""
    zero = False
    for i in range(len(sections) - 1, -1, -1):
        s = sections[i]
        if s == 0:
            zero = bool(out)
            continue
        if out and (zero or s < 1000):
            out += "零"
        zero = False
        out += _read_section(s) + ("", "万", "亿", "万亿")[i]
    if out.startswith("一十"):
        out = out[1:]
    if out.startswith("二") and len(out) > 1 and out[1] in "千万亿":
        out = "两" + out[1:]
    return out


def read_number(s: str) -> str:
    """`1,234.5` → 一千二百三十四点五. Digit by digit for a leading zero or 11+ digits
    (codes, phone numbers)."""
    s = s.replace(",", "")
    whole, _, frac = s.partition(".")
    if len(whole) >= 11 or (len(whole) > 1 and whole.startswith("0")):
        head = read_digits(whole)
    else:
        head = read_int(int(whole))
    return head + ("点" + read_digits(frac) if frac else "")


def read_year(s: str) -> str:
    return read_digits(s) if len(s) == 4 else read_number(s)


# ---- the scanner -----------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
_CUR = "|".join(re.escape(c) for c in CURRENCIES)

_SIGN = r"(?<![\dA-Za-z.,)])"
"""Where a - or + is a sign rather than an operator: not after a number, word or bracket."""

_RULES: list[tuple[str, str]] = [
    (
        "date",
        r"(?<![\d.,])(?P<y>\d{4})(?P<s>[-/.])(?P<mo>\d{1,2})(?P=date_s)(?P<d>\d{1,2})(?![\d.])",
    ),
    ("time", r"(?<![\d:])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?![\d:])"),
    (
        "range",
        rf"(?<![\d.,])(?P<a>{_NUM})\s*[-~～–—]\s*(?P<b>{_NUM})"
        rf"(?:(?P<y>(?=年))|\s*(?P<p>[%‰])|\s*(?P<u>{_UNIT})(?![A-Za-z]))?",
    ),
    ("year", r"(?<![\d.,])(?P<y>\d{4})(?=年)"),
    ("percent", rf"(?:{_SIGN}(?P<sign>[-+]))?(?<![\d.,])(?P<n>{_NUM})\s*(?P<p>[%‰])"),
    ("fraction", r"(?<![\d./])(?P<a>\d{1,3})/(?P<b>\d{1,3})(?![\d/])"),
    ("money", rf"(?P<c>{_CUR})\s*(?P<n>{_NUM})(?P<big>[万亿]?)"),
    ("unit", rf"(?:{_SIGN}(?P<sign>[-+]))?(?<![\d.,])(?P<n>{_NUM})\s*(?P<u>{_UNIT})(?![A-Za-z])"),
    ("signed", rf"{_SIGN}(?P<sign>[-+])(?P<n>{_NUM})"),
    ("number", rf"(?<![\d.,])(?P<n>{_NUM})"),
    ("hash", rf"#\s*(?P<n>{_NUM})"),
    ("versus", r"(?<![A-Za-z])(?:vs|VS|Vs)\.?(?![A-Za-z])"),
    ("abbr", r"(?<![A-Za-z])(?P<w>[A-Z]{2,6})(?![A-Za-z])"),
    ("op", r"(?<=\d)\s*(?P<o>[+=×÷*])\s*(?=[-\d])"),
    ("amp", r"\s*[&＆]\s*"),
    ("digits", r"\d+"),  # whatever the rules above leave (`.5`, `No.1`): digit by digit
    ("drop", "[" + re.escape("".join(sorted(DROPPED))) + "]"),
]


def _scoped(name: str, rx: str) -> str:
    """Group names are prefixed with the rule's so that rules can share them."""
    return rx.replace("(?P<", f"(?P<{name}_")


_SCAN = re.compile("|".join(f"(?P<{name}>{_scoped(name, rx)})" for name, rx in _RULES))
_OPS = {"+": "加", "=": "等于", "×": "乘", "*": "乘", "÷": "除以"}


def _fullwidth_to_ascii(text: str) -> str:
    """Fullwidth digits, letters and %: one character for one, so offsets are unchanged.
    Chinese punctuation (，。！) is left as it is."""
    out = []
    for c in text:
        o = ord(c)
        if 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A or c == "％":
            out.append(chr(o - 0xFEE0))
        else:
            out.append(c)
    return "".join(out)


def _reading(m: re.Match[str], scan: str) -> str:
    kind = m.lastgroup or ""

    groups = m.groupdict()

    def g(name: str) -> str:
        return groups.get(f"{kind}_{name}") or ""

    def has(name: str) -> bool:
        return groups.get(f"{kind}_{name}") is not None

    sign = "负" if has("sign") and g("sign") == "-" else "正" if has("sign") else ""
    if kind == "date":
        month, day = read_int(int(g("mo"))), read_int(int(g("d")))
        return read_digits(g("y")) + "年" + month + "月" + day + "日"
    if kind == "time":
        h, mi = int(g("h")), int(g("m"))
        hour = "两" if h == 2 else read_int(h)
        if mi == 0:
            return hour + "点"
        return hour + "点" + ("零" if mi < 10 else "") + read_int(mi) + "分"
    if kind == "range":
        if has("y"):
            return read_year(g("a")) + "到" + read_year(g("b"))
        if has("p"):
            per = "百分之" if g("p") == "%" else "千分之"
            return per + read_number(g("a")) + "到" + per + read_number(g("b"))
        unit = UNITS[g("u")] if has("u") else ""
        return read_number(g("a")) + "到" + read_number(g("b")) + unit
    if kind == "year":
        return read_year(g("y"))
    if kind == "percent":
        return sign + ("百分之" if g("p") == "%" else "千分之") + read_number(g("n"))
    if kind == "fraction":
        return read_int(int(g("b"))) + "分之" + read_int(int(g("a")))
    if kind == "money":
        return read_number(g("n")) + g("big") + CURRENCIES[g("c")]
    if kind == "unit":
        return sign + read_number(g("n")) + UNITS[g("u")]
    if kind == "signed":
        return sign + read_number(g("n"))
    before = scan[m.start() - 1] if m.start() > 0 else ""
    after = scan[m.end()] if m.end() < len(scan) else ""
    if kind == "number":
        n = g("n")
        if n == "2" and before != "第" and after in _LIANG_BEFORE:
            return "两"
        return read_number(n)
    if kind == "hash":
        return read_number(g("n")) + ("" if after == "号" else "号")
    if kind == "versus":
        return "对"
    if kind == "abbr":
        w = g("w")
        return w if w in SAID_AS_WORDS else " ".join(w)
    if kind == "op":
        return _OPS[g("o")]
    if kind == "digits":
        return read_digits(m.group())
    if kind == "amp":
        return "和"
    return ""  # drop


def normalize_for_tts(source: str) -> Normalized:
    """The text the engine should read for `source`, with the correspondence between the two."""
    scan = _fullwidth_to_ascii(source)
    out: list[str] = []
    pieces: list[Piece] = []
    pos = 0  # in the source
    length = 0  # of the output so far

    def emit(src_lo: int, src_hi: int, text: str) -> None:
        nonlocal length
        pieces.append(Piece(src_lo, src_hi, length, length + len(text)))
        out.append(text)
        length += len(text)

    for m in _SCAN.finditer(scan):
        if m.start() == m.end():
            continue
        for i in range(pos, m.start()):
            emit(i, i + 1, scan[i])
        emit(m.start(), m.end(), _reading(m, scan))
        pos = m.end()
    for i in range(pos, len(source)):
        emit(i, i + 1, scan[i])
    return Normalized(source=source, text="".join(out), pieces=tuple(pieces))
