"""Text normalization for speech: how numbers, English, units and symbols are read aloud. Pure.

Subtitles keep the script text; only the engine hears the normalized text. `normalize_for_speech`
therefore returns the spoken text *and* how it maps back to the original characters, so the
character timings the engine reports for the spoken text can be mapped onto the script
(`remap_timings`)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NORMALIZER_VERSION = 1
"""Bump when the reading of any input changes: it is part of the TTS cache key."""

_DIGITS = "零一二三四五六七八九"
_UNITS = ("", "十", "百", "千")
_BIG = ("", "万", "亿", "万亿")

# Longer spellings first so that "km/h" wins over "km", "ml" over "m".
_UNIT_READINGS: dict[str, str] = {
    "km/h": "千米每小时",
    "m/s": "米每秒",
    "kg": "千克",
    "mg": "毫克",
    "km": "公里",
    "cm": "厘米",
    "mm": "毫米",
    "ml": "毫升",
    "GB": "吉字节",
    "MB": "兆字节",
    "KB": "千字节",
    "TB": "太字节",
    "Hz": "赫兹",
    "kW": "千瓦",
    "g": "克",
    "m": "米",
    "L": "升",
    "h": "小时",
    "s": "秒",
    "℃": "摄氏度",
    "°C": "摄氏度",
    "°F": "华氏度",
}
_CURRENCY_BEFORE = {"$": "美元", "¥": "元", "￥": "元", "€": "欧元", "£": "英镑"}

# Read as a word rather than spelled out.
DEFAULT_LEXICON: dict[str, str] = {
    "NASA": "NASA",
    "NATO": "NATO",
    "IMAX": "IMAX",
    "OK": "OK",
}

_MEASURE_AFTER_TWO = "个人位条只天次张名件辆家间双"

_FULLWIDTH = {c: chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F) if chr(c - 0xFEE0).isalnum()}


def cn_int(n: int) -> str:
    """`1205` -> 一千二百零五, `10` -> 十, `100000` -> 十万. Reads up to 9999 万亿."""
    if n < 0:
        return "负" + cn_int(-n)
    if n == 0:
        return "零"
    if n >= 10**16:
        return cn_digits(str(n))
    groups: list[int] = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    pending_zero = False
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        if g == 0:
            pending_zero = bool(out)
            continue
        part = _group(g)
        if out and (pending_zero or g < 1000):
            out += "零"
        out += part + _BIG[gi]
        pending_zero = False
    return out[1:] if out.startswith("一十") else out


def _group(g: int) -> str:
    s = ""
    zero = False
    for pos in (3, 2, 1, 0):
        d = (g // 10**pos) % 10
        if d == 0:
            zero = bool(s)
            continue
        if zero:
            s += "零"
            zero = False
        s += _DIGITS[d] + _UNITS[pos]
    return s


def cn_digits(s: str) -> str:
    """Digit by digit: `2024` -> 二零二四 (years, codes, phone numbers)."""
    return "".join(_DIGITS[int(c)] for c in s if c.isdigit())


def _cn_decimal(whole: str, frac: str) -> str:
    return cn_int(int(whole)) + "点" + cn_digits(frac)


_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNIT_READINGS, key=len, reverse=True))
_TOKEN = re.compile(
    rf"""
    (?P<time>(?<![\d:])(?P<hh>[01]?\d|2[0-3]):(?P<mm>[0-5]\d)(?!\d))
  | (?P<cur>(?P<cs>[$¥￥€£])(?P<cn>\d[\d,]*(?:\.\d+)?))
  | (?P<ratio>(?<![\d.])(?P<ra>\d+)/(?P<rb>\d+)(?![\d/]))
  | (?P<num>(?P<sign>(?<![A-Za-z0-9])[-−])?(?P<int>\d[\d,]*|\d)(?:\.(?P<frac>\d+))?
        (?P<pct>%|％)?(?P<range>[~～](?P<int2>\d+))?
        (?P<unit>(?:{_UNIT_RE})(?![A-Za-z]))?)
  | (?P<eng>[A-Za-z]+(?:[.'][A-Za-z]+)*)
  | (?P<sym>[&+=×÷<>@#*_`~^|\\])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Piece:
    """The original text `text[lo:hi]` is spoken as `spoken` (equal to it when untouched)."""

    lo: int
    hi: int
    spoken: str


@dataclass(frozen=True)
class SpeechText:
    spoken: str
    pieces: tuple[Piece, ...]
    """Cover the original text end to end, in order."""


def normalize_for_speech(text: str, lexicon: Mapping[str, str] | None = None) -> SpeechText:
    """The text as it should be spoken, plus the mapping to `text`'s characters.

    Numbers (integers, decimals, percentages, ranges, years, times, ratios), units, currency,
    all-caps acronyms (spelled letter by letter unless in `lexicon`) and a few symbols are
    rewritten; everything else is passed through. NFKC-style folding of full-width digits and
    letters is done per character so offsets are never shifted."""
    lex = {**DEFAULT_LEXICON, **(lexicon or {})}
    folded = text.translate(_FULLWIDTH)
    pieces: list[Piece] = []
    pos = 0

    for m in _TOKEN.finditer(folded):
        spoken = _read(m, folded, lex)
        if spoken is None:
            continue
        if m.start() > pos:
            pieces.append(Piece(pos, m.start(), text[pos : m.start()]))
        pieces.append(Piece(m.start(), m.end(), spoken))
        pos = m.end()
    if pos < len(text):
        pieces.append(Piece(pos, len(text), text[pos:]))
    return SpeechText("".join(p.spoken for p in pieces), tuple(pieces))


def _read(m: re.Match[str], s: str, lex: Mapping[str, str]) -> str | None:
    if m.group("time"):
        h, mi = int(m.group("hh")), int(m.group("mm"))
        return cn_int(h) + "点" + ("" if mi == 0 else ("零" if mi < 10 else "") + cn_int(mi) + "分")
    if m.group("cur"):
        n = m.group("cn").replace(",", "")
        whole, _, frac = n.partition(".")
        read = _cn_decimal(whole, frac) if frac else cn_int(int(whole))
        return read + _CURRENCY_BEFORE[m.group("cs")]
    if m.group("ratio"):
        a, b = int(m.group("ra")), int(m.group("rb"))
        if b == 0 or a > b * 1000:
            return None  # not a fraction (a date-like 12/31 is read as numbers + slash)
        return cn_int(b) + "分之" + cn_int(a)
    if m.group("num"):
        return _read_number(m, s)
    if m.group("eng"):
        word = m.group("eng")
        if word in lex:
            return lex[word]
        letters = word.replace(".", "")
        if len(letters) >= 2 and letters.isupper() and letters.isalpha() and len(letters) <= 6:
            return " ".join(letters)
        return None
    sym = m.group("sym")
    if sym is None:
        return None
    return {
        "&": "和",
        "+": "加",
        "=": "等于",
        "×": "乘",
        "÷": "除以",
        "<": "小于",
        ">": "大于",
        "@": "艾特",
    }.get(sym, "")


def _read_number(m: re.Match[str], s: str) -> str | None:
    whole = m.group("int").replace(",", "")
    frac = m.group("frac")
    pct = m.group("pct")
    unit = m.group("unit")
    sign = "负" if m.group("sign") else ""
    after = s[m.end() : m.end() + 1]
    is_year = (
        not frac
        and not pct
        and not unit
        and len(whole) == 4
        and not m.group("sign")
        and after == "年"
    )
    if is_year or (len(whole) > 1 and whole.startswith("0") and not frac) or len(whole) > 11:
        body = cn_digits(whole)
        if frac:
            body += "点" + cn_digits(frac)
    elif frac:
        body = _cn_decimal(whole, frac)
    else:
        body = cn_int(int(whole))
        if whole == "2" and not pct and not unit and after and after in _MEASURE_AFTER_TWO:
            body = "两"
    if m.group("range"):
        hi = m.group("int2")
        body += "到" + cn_int(int(hi))
    if pct:
        return sign + "百分之" + body
    if unit:
        return sign + body + _UNIT_READINGS[unit]
    return sign + body


def remap_timings(
    pieces: Sequence[Piece],
    spoken_timings: Sequence[tuple[int, int]],
    n_original: int,
) -> list[tuple[int, int]]:
    """One `(start_ms, end_ms)` per character of the original text, from the timings of the
    spoken text. A piece's span is shared evenly among the original characters it replaced;
    an original character that is not spoken at all gets an empty interval where the previous
    one ended. `spoken_timings == []` (engine gave none) maps to `[]`."""
    if not spoken_timings:
        return []
    out: list[tuple[int, int]] = []
    s_pos = 0
    prev_end = 0
    for p in pieces:
        n_orig = p.hi - p.lo
        n_spoken = len(p.spoken)
        chunk = spoken_timings[s_pos : s_pos + n_spoken]
        s_pos += n_spoken
        if not chunk:
            out.extend([(prev_end, prev_end)] * n_orig)
            continue
        if n_orig == n_spoken:
            out.extend(chunk)
        else:
            a, b = chunk[0][0], chunk[-1][1]
            for k in range(n_orig):
                out.append((a + (b - a) * k // n_orig, a + (b - a) * (k + 1) // n_orig))
        prev_end = out[-1][1]
    if len(out) != n_original:
        raise ValueError("pieces do not cover the original text")
    return out
