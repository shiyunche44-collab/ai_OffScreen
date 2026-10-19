"""Text normalization before speech: how digits, units, symbols and Latin letters in a Chinese
narration are read (ARCHITECTURE §7.3). Pure.

`to_spoken(text)` gives what the engine should read ("2008年" -> "二零零八年", "50%" ->
"百分之五十", "3.5km" -> "三点五公里", "FBI" -> "F B I") together with the pieces it was made
of, so that timestamps the engine reports for the spoken text map back onto the characters of
the original (`Spoken.source_timings`): captions and plans keep showing what the script says.

Everything not covered by a rule is passed through unchanged, so a plain CJK text is its own
spoken form. Readings are a best guess from the surrounding characters; what they cannot tell
apart (a score "3:2" from a time "3:20", a duration "2年" from a year) follows the more common
use in narration."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DIGITS = "零一二三四五六七八九"

_FULL_WIDTH = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)} | {0x3000: 0x20}
"""Full-width ASCII (０-９, Ａ-Ｚ, ％, ．, ＄ ...) to half-width: one character each, so the
pieces stay aligned with the original."""

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

UNITS: dict[str, str] = {
    "km/h": "公里每小时",
    "mph": "英里每小时",
    "km": "公里",
    "cm": "厘米",
    "mm": "毫米",
    "kg": "公斤",
    "mg": "毫克",
    "ml": "毫升",
    "mL": "毫升",
    "min": "分钟",
    "m": "米",
    "g": "克",
    "t": "吨",
    "L": "升",
    "h": "小时",
    "s": "秒",
    "°C": "摄氏度",
    "℃": "摄氏度",
    "°F": "华氏度",
    "℉": "华氏度",
    "°": "度",
}
"""Units read after a number; Latin ones only when no letter follows ("5min" is minutes, "5mb"
is left alone)."""

CURRENCIES: dict[str, str] = {"$": "美元", "¥": "元", "￥": "元", "€": "欧元", "£": "英镑"}

MEASURES = [
    "小时",
    "分钟",
    "个",
    "只",
    "位",
    "名",
    "人",
    "次",
    "天",
    "年",
    "周",
    "条",
    "件",
    "种",
    "部",
    "本",
    "张",
    "岁",
    "倍",
    "把",
    "座",
    "辆",
    "家",
    "头",
    "匹",
    "项",
    "场",
    "点",
    "层",
    "句",
    "段",
    "架",
    "艘",
    "颗",
    "根",
    "双",
    "份",
    "台",
    "款",
    "门",
    "枪",
    "拳",
    "刀",
    "步",
    "遍",
    "秒",
    "米",
    "公里",
    "公斤",
    "斤",
    "吨",
    "克",
    "千",
    "万",
    "亿",
    "百",
]
"""Measure words after which a lone 2 is 两 ("2个人" -> "两个人"), unless it is an ordinal
("第2次")."""

WORD_ACRONYMS = frozenset(
    [
        "NASA",
        "NATO",
        "OPEC",
        "UNESCO",
        "UNICEF",
        "AIDS",
        "COVID",
        "SARS",
        "LASER",
        "RADAR",
        "SWAT",
        "ISIS",
        "FIFA",
    ]
)
"""Capitalized abbreviations said as a word; any other run of 2-5 capitals is spelled out."""

_RANGE_SEP = r"\s*[-~～–—至到]\s*"
_MEASURE = "|".join(MEASURES)


@dataclass(frozen=True)
class Piece:
    """`source[src_lo:src_hi]` is read as `text[out_lo:out_hi]`."""

    src_lo: int
    src_hi: int
    out_lo: int
    out_hi: int


@dataclass(frozen=True)
class Spoken:
    source: str
    text: str
    """What to give the engine."""
    pieces: tuple[Piece, ...]
    """Cover `source` and `text` in order, without gaps."""

    def source_timings(self, timings: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Per-character timings of the spoken text -> per-character timings of the source.

        A piece read with as many characters as it has is mapped one to one; otherwise the
        time its reading takes is divided evenly among its source characters, and a piece read
        as nothing (a dropped hyphen) gets an empty interval where the previous one ended.
        Empty timings (an engine without timestamps) stay empty."""
        if not timings:
            return []
        if len(timings) != len(self.text):
            raise ValueError("timings must have one entry per character of the spoken text")
        out: list[tuple[int, int]] = []
        prev_end = 0
        for p in self.pieces:
            n_src, span = p.src_hi - p.src_lo, timings[p.out_lo : p.out_hi]
            if n_src == len(span):
                pairs = list(span)
            elif not span:
                pairs = [(prev_end, prev_end)] * n_src
            else:
                voiced = [t for t in span if t[1] > t[0]] or list(span)
                a, b = min(s for s, _ in voiced), max(e for _, e in voiced)
                pairs = [
                    (round(a + (b - a) * k / n_src), round(a + (b - a) * (k + 1) / n_src))
                    for k in range(n_src)
                ]
            for s, e in pairs:
                s = max(s, prev_end)
                e = max(e, s)
                out.append((s, e))
                prev_end = e
        return out


# ---- readings ----------------------------------------------------------------------------
def read_digits(digits: str) -> str:
    """One by one: "2008" -> "二零零八"."""
    return "".join(DIGITS[int(d)] for d in digits)


def read_integer(digits: str) -> str:
    """As a quantity: "1500" -> "一千五百", "10086" -> "一万零八十六", "12" -> "十二". More
    than 16 digits, or a leading zero ("007"), is read digit by digit."""
    digits = digits.lstrip("+")
    if len(digits) > 16 or (len(digits) > 1 and digits[0] == "0"):
        return read_digits(digits)
    n = int(digits)
    if n == 0:
        return "零"
    groups: list[int] = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    zero = False
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            zero = bool(out)
            continue
        if out and (zero or g < 1000):
            out += "零"
        out += _read_group(g, lead=not out, before_unit=i > 0)
        out += ("", "万", "亿", "万亿")[i]
        zero = False
    return out


def _read_group(g: int, *, lead: bool, before_unit: bool) -> str:
    if g == 2 and before_unit:
        return "两"  # 两万, 两亿
    out = ""
    pending_zero = False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = g // value % 10
        if d == 0:
            pending_zero = bool(out)
            continue
        if pending_zero:
            out += "零"
            pending_zero = False
        if unit == "千" and d == 2:
            out += "两千"
        elif unit == "十" and d == 1 and lead and not out:
            out += "十"  # 十二, not 一十二
        else:
            out += DIGITS[d] + unit
    return out


def read_number(num: str) -> str:
    """A number as written in text: thousands separators, decimals ("3.14" -> "三点一四")."""
    num = num.replace(",", "")
    whole, _, frac = num.partition(".")
    out = read_integer(whole)
    return out + "点" + read_digits(frac) if frac else out


def _read_clock(hour: str, minute: str) -> str:
    out = "两" if hour.lstrip("0") == "2" else read_integer(hour.lstrip("0") or "0")
    out += "点"
    if minute != "00":
        out += ("零" if minute[0] == "0" else "") + read_integer(minute.lstrip("0")) + "分"
    return out


def _read_in_token(digits: str) -> str:
    """Digits inside a model name or code ("R2-D2", "T-800", "MP5"): short ones as a number."""
    return read_digits(digits) if len(digits) > 2 else read_integer(digits)


# ---- rules -------------------------------------------------------------------------------
Rule = tuple[str, Callable[[re.Match[str]], str]]

_RULES: list[Rule] = [
    # 10:30 -> 十点三十分; 10:05 -> 十点零五分; 2:00 -> 两点
    (r"(?<!\d)([01]?\d|2[0-4]):([0-5]\d)(?![\d:])", lambda m: _read_clock(m[1], m[2])),
    # 3:2 -> 三比二
    (
        r"(?<!\d)(\d{1,3}):(\d{1,3})(?![\d:])",
        lambda m: read_integer(m[1]) + "比" + read_integer(m[2]),
    ),
    # $5亿 -> 五亿美元
    (
        rf"([$¥￥€£])({_NUM})([万亿千百]*)",
        lambda m: read_number(m[2]) + m[3] + CURRENCIES[m[1]],
    ),
    # 10%-20% / 10-20% -> 百分之十到百分之二十
    (
        rf"({_NUM})%?{_RANGE_SEP}({_NUM})[%％]",
        lambda m: "百分之" + read_number(m[1]) + "到百分之" + read_number(m[2]),
    ),
    (rf"({_NUM})[%％]", lambda m: "百分之" + read_number(m[1])),
    (rf"({_NUM})‰", lambda m: "千分之" + read_number(m[1])),
    # 1990-2000年 / 2023-2024赛季 -> 一九九零到二零零零年; 2008年 -> 二零零八年
    (
        rf"(?<![\d.])(\d{{4}}){_RANGE_SEP}(\d{{4}})(?=年|赛季)"
        rf"|(?<![\d.])((?:19|20)\d\d){_RANGE_SEP}((?:19|20)\d\d)(?![\d.]|{_MEASURE})",
        lambda m: read_digits(m[1] or m[3]) + "到" + read_digits(m[2] or m[4]),
    ),
    (r"(?<![\d.])(\d{4})(?=年|赛季)", lambda m: read_digits(m[1])),
    # 1990s -> 一九九零年代
    (r"(?<!\d)(\d{3}0)s(?![A-Za-z])", lambda m: read_digits(m[1]) + "年代"),
    # 3/4 -> 四分之三 (not dates like 2024/10/19)
    (
        r"(?<![\d/])(\d{1,4})/(\d{1,4})(?![\d/])",
        lambda m: read_integer(m[2]) + "分之" + read_integer(m[1]),
    ),
    # 3-5 -> 三到五 (the unit, if any, follows on its own)
    (
        rf"(?<![\d.])({_NUM}){_RANGE_SEP}({_NUM})(?![\d.])",
        lambda m: read_number(m[1]) + "到" + read_number(m[2]),
    ),
    # 3.5km -> 三点五公里; 37℃ -> 三十七摄氏度
    (
        rf"({_NUM})\s?("
        + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
        + ")(?![A-Za-z])",
        lambda m: read_number(m[1]) + UNITS[m[2]],
    ),
    # R2-D2 -> R二D二; T-800 -> T八零零; 4K -> 四K
    (
        r"(?=[A-Za-z0-9-]*\d)[A-Za-z]+(?:-?[A-Za-z0-9]+)+|\d+[A-Za-z]+(?:-?[A-Za-z0-9]+)*",
        lambda m: re.sub(r"-?(\d+)", lambda d: _read_in_token(d[1]), m[0]).replace("-", ""),
    ),
    # 2个 -> 两个 (第2个 stays 第二个)
    (
        rf"(?<![\d.第])2(?![\d.])(?={_MEASURE})",
        lambda m: "两",
    ),
    (rf"(?<![\dA-Za-z])[-−](?={_NUM})", lambda m: "负"),
    (rf"#(?={_NUM})", lambda m: "第"),
    (_NUM, lambda m: read_number(m[0])),
    # FBI -> F B I; NASA stays
    (
        r"(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])",
        lambda m: m[1] if m[1] in WORD_ACRONYMS else " ".join(m[1]),
    ),
    (r"(?<=\d)\s*\+\s*(?=\d)", lambda m: "加"),
    (r"(?<=\d)\s*=\s*(?=\d)", lambda m: "等于"),
    (r"(?<=\d)\s*[×xX*]\s*(?=\d)", lambda m: "乘"),
    (r"(?<=\d)\s*÷\s*(?=\d)", lambda m: "除以"),
    (r"\s*&\s*", lambda m: "和"),
]

_PATTERNS = [re.compile(p) for p, _ in _RULES]
_MASTER = re.compile("|".join(f"(?P<r{i}>{p})" for i, (p, _) in enumerate(_RULES)))


def to_spoken(text: str) -> Spoken:
    """What the engine should read for `text`, and how it lines up with it."""
    work = text.translate(_FULL_WIDTH)
    out: list[str] = []
    pieces: list[Piece] = []
    pos = n_out = 0

    def emit(src_lo: int, src_hi: int, said: str) -> None:
        nonlocal n_out
        out.append(said)
        pieces.append(Piece(src_lo, src_hi, n_out, n_out + len(said)))
        n_out += len(said)

    for m in _MASTER.finditer(work):
        if not m[0]:
            continue
        if m.start() > pos:
            emit(pos, m.start(), text[pos : m.start()])
        assert m.lastgroup is not None
        i = int(m.lastgroup[1:])
        # the winning rule on its own, so that its groups are numbered from 1
        sub = _PATTERNS[i].match(work, m.start())
        assert sub is not None and sub.end() == m.end()
        emit(m.start(), m.end(), _RULES[i][1](sub))
        pos = m.end()
    if pos < len(text):
        emit(pos, len(text), text[pos:])
    return Spoken(text, "".join(out), tuple(pieces))
