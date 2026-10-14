"""Text normalization before TTS: how digits, units and symbols are read aloud. Pure.

`to_spoken` rewrites a narration text into what the engine should be given ("50%" ->
"百分之五十") and keeps, for every source character, the span of spoken characters that replaced
it, so the engine's per-character timings can be mapped back onto the original text
(`SpokenText.remap_timings`): the plan, subtitles and highlighting all stay in terms of the
text the user wrote.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_DIGITS = "零一二三四五六七八九"
_SMALL_UNITS = ("", "十", "百", "千")
_BIG_UNITS = ("", "万", "亿", "兆")
_MAX_CARDINAL = 10**16
_DIGITWISE_LEN = 8
"""A plain integer of this many digits or more (phone numbers, IDs) is read digit by digit."""

_CURRENCY = {"$": "美元", "¥": "元", "￥": "元", "€": "欧元", "£": "英镑"}
_UNITS = {
    "km/h": "公里每小时",
    "km": "公里",
    "kg": "公斤",
    "mg": "毫克",
    "g": "克",
    "cm": "厘米",
    "mm": "毫米",
    "ml": "毫升",
    "mL": "毫升",
    "L": "升",
    "m²": "平方米",
    "m2": "平方米",
    "㎡": "平方米",
    "m": "米",
    "min": "分钟",
    "h": "小时",
    "℃": "摄氏度",
    "°C": "摄氏度",
    "℉": "华氏度",
    "°F": "华氏度",
    "%": "%",
    "‰": "‰",
}
"""Units read right after a number. `s` is left out on purpose: "1990s" is not seconds."""
_MEASURE_WORDS = "个位名次条只张天间场点倍件项套种"
"""A lone 2 before one of these is read 两."""
_SYMBOLS = {"&": "和", "×": "乘", "÷": "除以", "=": "等于", "≈": "约等于"}
_MARKUP = "*_#`~"


def _four_digits(n: int) -> str:
    out, pending_zero = "", False
    for pos in (3, 2, 1, 0):
        d = n // 10**pos % 10
        if d == 0:
            pending_zero = pending_zero or bool(out)
            continue
        if pending_zero:
            out += "零"
            pending_zero = False
        out += _DIGITS[d] + _SMALL_UNITS[pos]
    return out


def cardinal(n: int) -> str:
    """`n` as spoken Chinese: 105 -> 一百零五, 12 -> 十二. 10**16 or more is read digit by digit."""
    if n < 0:
        raise ValueError("cardinal takes a non-negative number")
    if n == 0:
        return "零"
    if n >= _MAX_CARDINAL:
        return digits(str(n))
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out, pending_zero = "", False
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            pending_zero = pending_zero or bool(out)
            continue
        if out and (pending_zero or g < 1000):
            out += "零"
        pending_zero = False
        out += _four_digits(g) + _BIG_UNITS[idx]
    return out.removeprefix("一") if out.startswith("一十") else out


def digits(s: str) -> str:
    """Each digit read on its own: "2024" -> 二零二四."""
    return "".join(_DIGITS[int(c)] for c in s if c.isdigit())


def _two_to_liang(spoken: str, next_char: str) -> str:
    """二 -> 两 where speech wants it: 两千 / 两万 / 两亿, a leading 两百, a lone 两个."""
    if spoken == "二":
        return "两" if next_char and next_char in _MEASURE_WORDS else spoken
    out = re.sub(r"二(?=[千万亿])", "两", spoken)
    return re.sub(r"^二(?=百)", "两", out)


def _number(int_part: str, dec_parts: Sequence[str], next_char: str) -> str:
    """A plain number: cardinal with an optional decimal ("3.14" -> 三点一四), digit by digit
    for leading zeros / IDs / version-like "1.2.3"."""
    if len(dec_parts) > 1:  # 1.2.3
        return "点".join([digits(int_part), *(digits(d) for d in dec_parts)])
    if len(int_part) > 1 and int_part.startswith("0") and not dec_parts:
        return digits(int_part)
    if not dec_parts and len(int_part) >= _DIGITWISE_LEN:
        return digits(int_part)
    whole = _two_to_liang(cardinal(int(int_part)), "" if dec_parts else next_char)
    if dec_parts:
        whole += "点" + digits(dec_parts[0])
    return whole


def _time(hour: str, minute: str) -> str:
    h = _two_to_liang(cardinal(int(hour)), "点")
    m = int(minute)
    if m == 0:
        return h + "点"
    return h + "点" + ("零" if m < 10 else "") + cardinal(m) + "分"


@dataclass(frozen=True)
class SpokenText:
    text: str
    """What the engine is given."""
    pieces: tuple[tuple[int, int, int, int], ...]
    """`(src_lo, src_hi, spoken_lo, spoken_hi)` covering every source character, in order."""

    @property
    def source_len(self) -> int:
        return self.pieces[-1][1] if self.pieces else 0

    def remap_timings(self, timings: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Engine timings (one per spoken character) -> one `(start_ms, end_ms)` per source
        character. A replaced span's time is shared evenly by the source characters it
        replaced; dropped characters (markup) get an empty interval. Monotonic."""
        if not timings:
            return []
        if len(timings) != len(self.text):
            raise ValueError(f"{len(timings)} timings for {len(self.text)} spoken characters")
        out: list[tuple[int, int]] = []
        prev_end = 0
        for src_lo, src_hi, sp_lo, sp_hi in self.pieces:
            n_src = src_hi - src_lo
            if sp_hi - sp_lo == n_src:
                spans = [timings[sp_lo + k] for k in range(n_src)]
            elif sp_hi == sp_lo:
                spans = [(prev_end, prev_end)] * n_src
            else:
                a, b = timings[sp_lo][0], timings[sp_hi - 1][1]
                spans = [
                    (round(a + (b - a) * k / n_src), round(a + (b - a) * (k + 1) / n_src))
                    for k in range(n_src)
                ]
            for start, end in spans:
                s = max(start, prev_end)
                e = max(end, s)
                out.append((s, e))
                prev_end = e
        return out


_FULLWIDTH = str.maketrans("０１２３４５６７８９％", "0123456789%")
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_NUMBER_RE = (
    r"(?P<cur>[$¥￥€£])?"
    r"(?P<neg>(?<![A-Za-z0-9)）])[-−])?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<dec>(?:\.\d+)*)"
    r"(?P<mag>[万亿]{1,2})?"
    rf"(?:\s?(?P<unit>{_UNIT_ALT})(?![A-Za-z]))?"
)
_BASE_PATTERN = (
    r"(?P<frac>(?<![\d/])\d+/\d+(?![\d/]))"
    r"|(?P<time>(?<![\d:])(?P<hh>[01]?\d|2[0-3]):(?P<mm>[0-5]\d)(?![\d:]))"
    r"|(?P<range>(?<=\d)[~～](?=\d))"
    rf"|{_NUMBER_RE}"
    r"|(?P<sym>[&×÷=≈])"
    r"|(?P<markup>[*_#`~]+)"
)


def _compile(lexicon: Mapping[str, str]) -> re.Pattern[str]:
    if not lexicon:
        return re.compile(_BASE_PATTERN)
    words = sorted(lexicon, key=len, reverse=True)
    alts = []
    for w in words:
        pat = re.escape(w)
        if w[0].isascii() and w[0].isalnum():
            pat = r"(?<![A-Za-z0-9])" + pat
        if w[-1].isascii() and w[-1].isalnum():
            pat += r"(?![A-Za-z0-9])"
        alts.append(pat)
    return re.compile(rf"(?P<lex>{'|'.join(alts)})|{_BASE_PATTERN}")


def _read_number(m: re.Match[str], text: str) -> str:
    int_part = m["int"].replace(",", "")
    dec_parts = [d for d in m["dec"].split(".") if d]
    nxt = text[m.end() : m.end() + 1]
    unit = m["unit"]
    mag = m["mag"] or ""
    plain = not dec_parts and "," not in m["int"] and not mag
    if plain and len(int_part) == 4 and re.match(r"\s?年", text[m.end() :]):
        body = digits(int_part)  # 2024年: a year is read digit by digit
    else:
        body = _number(int_part, dec_parts, nxt if not unit else "")
    if mag:
        body += mag
    neg = m["neg"]
    if unit in ("%", "‰"):
        return ("负" if neg else "") + ("百分之" if unit == "%" else "千分之") + body
    cold = unit in ("℃", "°C", "℉", "°F")
    prefix = ("零下" if cold else "负") if neg else ""
    if unit:
        body += _UNITS[unit]
    if m["cur"]:
        body += _CURRENCY[m["cur"]]
    return prefix + body


def to_spoken(text: str, lexicon: Mapping[str, str] | None = None) -> SpokenText:
    """Rewrite `text` for the engine. `lexicon` maps written forms to the reading to give
    the engine (English names, characters with several readings); it wins over every rule."""
    text = text.translate(_FULLWIDTH)
    pattern = _compile(lexicon or {})
    out: list[str] = []
    pieces: list[tuple[int, int, int, int]] = []
    spoken_len = 0
    pos = 0

    def emit(src_lo: int, src_hi: int, spoken: str) -> None:
        nonlocal spoken_len
        pieces.append((src_lo, src_hi, spoken_len, spoken_len + len(spoken)))
        out.append(spoken)
        spoken_len += len(spoken)

    for m in pattern.finditer(text):
        if m.start() == m.end():
            continue
        if m.start() > pos:
            emit(pos, m.start(), text[pos : m.start()])
        if lexicon and m["lex"] is not None:
            spoken = lexicon[m["lex"]]
        elif m["frac"] is not None:
            a, b = m["frac"].split("/")
            spoken = f"{_number(b, [], '')}分之{_number(a, [], '')}"
        elif m["time"] is not None:
            spoken = _time(m["hh"], m["mm"])
        elif m["range"] is not None:
            spoken = "到"
        elif m["sym"] is not None:
            spoken = _SYMBOLS[m["sym"]]
        elif m["markup"] is not None:
            spoken = ""
        else:
            spoken = _read_number(m, text)
        emit(m.start(), m.end(), spoken)
        pos = m.end()
    if pos < len(text):
        emit(pos, len(text), text[pos:])
    return SpokenText("".join(out), tuple(pieces))
