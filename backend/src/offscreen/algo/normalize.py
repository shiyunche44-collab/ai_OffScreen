"""Text normalization before speech synthesis (ARCHITECTURE §7.3). Pure.

Numbers, units and symbols are rewritten the way a narrator reads them ("3.5kg" -> "三点五千克").
The result keeps a map from every spoken character back to the characters of the original, so
engine timestamps can be put back on the text the script actually contains."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from offscreen.algo.tts import WordSpan, char_timings_from_spans

_DIGITS = "零一二三四五六七八九"
_SMALL_UNITS = ["", "十", "百", "千"]
_BIG_UNITS = ["", "万", "亿", "万亿"]

_UNITS = {
    "km/h": "千米每小时",
    "km²": "平方千米",
    "m²": "平方米",
    "kg": "千克",
    "km": "千米",
    "cm": "厘米",
    "mm": "毫米",
    "mg": "毫克",
    "ml": "毫升",
    "mL": "毫升",
    "min": "分钟",
    "g": "克",
    "m": "米",
    "L": "升",
    "℃": "摄氏度",
    "°C": "摄氏度",
    "℉": "华氏度",
    "°": "度",
}
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))

_SYMBOLS = {
    "&": "和",
    "+": "加",
    "=": "等于",
    "×": "乘",
    "÷": "除以",
    "≥": "大于等于",
    "≤": "小于等于",
}
# Characters that are markup or decoration, never spoken.
_STRIP = re.compile("[*#_`​‌‍﻿\U0001f300-\U0001faff☀-➿️]")
# Before these, 2 is read 两 (2个人, 2天) rather than 二.
_LIANG_AFTER = "个人天只条名位次倍张件头匹辆部本家"

_TOKEN = re.compile(
    r"(?P<time>(?<![\d.:])\d{1,2}:\d{2}(?![\d:]))"
    r"|(?P<range>(?<=\d)[-~～–—](?=\d))"
    r"|(?P<percent>(?<![A-Za-z\d])(?P<ptail>-?\d[\d,]*(?:\.\d+)?)%)"
    r"|(?P<frac>(?<![A-Za-z\d./])\d+/\d+(?![\d/]))"
    r"|(?P<year>(?<![A-Za-z\d])\d{4}(?=年))"
    r"|(?P<num>(?<![A-Za-z\d.])(?:(?<![A-Za-z\d])-)?\d[\d,]*(?:\.\d+)?)"
    r"(?:(?P<unit>" + _UNIT_ALT + r")(?![A-Za-z]))?"
    r"|(?P<strip>" + _STRIP.pattern + ")"
    r"|(?P<sym>[&+=×÷≥≤])"
    r"|(?P<dots>\.{3,}|。{3,})"
)


def int_to_cn(n: int) -> str:
    """Cardinal reading of a non-negative integer: 1024 -> 一千零二十四, 10 -> 十."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return "零"
    groups: list[int] = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    zero_pending = False
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        if g == 0:
            zero_pending = bool(out)
            continue
        part = _group_cn(g)
        # A group below 1000 after a higher one needs a 零 (10001 -> 一万零一).
        if out and (zero_pending or g < 1000):
            out += "零"
        zero_pending = False
        big = _BIG_UNITS[gi] if gi < 3 else "万亿"
        out += part + big
    if out.startswith("一十"):
        out = out[1:]
    return out


def _group_cn(g: int) -> str:
    out = ""
    zero = False
    for pos in range(3, -1, -1):
        d = (g // 10**pos) % 10
        if d == 0:
            zero = bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        out += _DIGITS[d] + _SMALL_UNITS[pos]
    return out


def _digits_cn(s: str) -> str:
    return "".join(_DIGITS[int(c)] for c in s if c.isdigit())


def _number_cn(raw: str, *, liang_next: str = "") -> str:
    neg = raw.startswith("-")
    body = raw.lstrip("-").replace(",", "")
    int_part, _, frac = body.partition(".")
    if (len(int_part) > 1 and int_part[0] == "0") or len(int_part) >= 9:
        spoken = _digits_cn(int_part)  # ids, phone numbers
    else:
        value = int(int_part)
        spoken = (
            "两" if value == 2 and not frac and liang_next and liang_next in _LIANG_AFTER else ""
        )
        spoken = spoken or int_to_cn(value)
    if frac:
        spoken += "点" + _digits_cn(frac)
    return ("负" if neg else "") + spoken


@dataclass(frozen=True)
class Normalized:
    text: str
    """What is sent to the engine."""
    origin: tuple[tuple[int, int], ...]
    """For each character of `text`, the `[lo, hi)` range of original characters it speaks."""


def normalize_for_speech(text: str) -> Normalized:
    out: list[str] = []
    origin: list[tuple[int, int]] = []

    def emit(spoken: str, lo: int, hi: int) -> None:
        m, span = len(spoken), hi - lo
        for j, ch in enumerate(spoken):
            a = lo + (j * span) // m
            b = max(a + 1, lo + ((j + 1) * span) // m) if span else lo
            out.append(ch)
            origin.append((min(a, hi), min(b, hi)))

    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() > pos:
            emit(text[pos : m.start()], pos, m.start())
        pos = m.end()
        lo, hi = m.span()
        if m.group("time"):
            h, mi = m.group("time").split(":")
            spoken = (
                int_to_cn(int(h))
                + "点"
                + ("" if mi == "00" else ("零" if mi[0] == "0" else "") + int_to_cn(int(mi)) + "分")
            )
            spoken = spoken.replace("点零零", "点")
        elif m.group("range"):
            spoken = "到"
        elif m.group("percent"):
            spoken = "百分之" + _number_cn(m.group("ptail"))
        elif m.group("frac"):
            a, b = m.group("frac").split("/")
            spoken = int_to_cn(int(b)) + "分之" + int_to_cn(int(a)) if int(b) else m.group(0)
        elif m.group("year"):
            spoken = _digits_cn(m.group("year"))
        elif m.group("num"):
            unit = m.group("unit")
            nxt = text[hi : hi + 1]
            spoken = _number_cn(m.group("num"), liang_next=nxt)
            if unit:
                spoken += _UNITS[unit]
        elif m.group("strip"):
            continue
        elif m.group("sym"):
            spoken = _SYMBOLS[m.group("sym")]
        elif m.group("dots"):
            spoken = "…"
        else:  # pragma: no cover - the alternatives above are exhaustive
            spoken = m.group(0)
        if spoken:
            emit(spoken, lo, hi)
    if pos < len(text):
        emit(text[pos:], pos, len(text))
    return Normalized("".join(out), tuple(origin))


def remap_timings(
    norm: Normalized, spoken_timings: Sequence[tuple[int, int]], n_orig: int, duration_ms: int
) -> list[tuple[int, int]]:
    """Engine timings for `norm.text` -> one `(start_ms, end_ms)` per original character."""
    if not spoken_timings:
        return []
    spans = [
        WordSpan(float(s), float(e), lo, hi)
        for (s, e), (lo, hi) in zip(spoken_timings, norm.origin, strict=True)
    ]
    return char_timings_from_spans(n_orig, spans, duration_ms)
