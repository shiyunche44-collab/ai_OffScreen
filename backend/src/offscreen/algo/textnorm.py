"""Text normalization before speech synthesis: how digits, symbols, units and a few Latin
abbreviations are read in Chinese narration. Pure.

The engine speaks `SpeechText.text`; subtitles keep showing the script text. The two are tied by
`pieces` (a monotonic cover of both strings), so timestamps the engine reports for the spoken text
map back onto the characters of the source (`SpeechText.source_timings`).

What is handled (ARCHITECTURE §7.3):
- integers by value (`1024` → 一千零二十四, `10,000` → 一万), with 两 for a lone 2 before a
  measure word and for 2千 / 2万 / 2亿; decimals (`3.14` → 三点一四); negatives (`-5` → 负五)
- digit by digit: years before 年 (`1990年` → 一九九零年), numbers glued to Latin letters
  (`B612`), leading zeros (`007`) and runs of 11 or more digits (phone numbers)
- percentages (`50%` → 百分之五十), fractions (`3/4` → 四分之三), dates (`2024-10-13`), clock
  times (`10:05` → 十点零五分), ranges (`3-5` → 三到五; `3-2` with the left side not smaller →
  三比二, a score), scores (`2:1` → 二比一), arithmetic between numbers (`+ × ÷ =`)
- currency signs (`$100` → 一百美元, `¥1.5亿` → 一点五亿元) and units after a number (`km/h`,
  `kg`, `cm`, `℃`, `°` …)
- `&` → 和; `vs` → 对; `Mr.` / `Mrs.` / `Ms.` / `Dr.` spelled out

Everything else is passed through: Latin words and acronyms are left to the engine, which reads
them itself; punctuation stays because engines pause on it."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

VERSION = 1
"""Bump when the spoken form of some text changes (it is part of what the voice-over says)."""

_DIGITS = "零一二三四五六七八九"
_GROUP_UNITS = ("", "万", "亿", "万亿")

_MEASURE_WORDS = (
    "小时|分钟|公里|千米|公斤|千克|个|位|次|人|只|条|张|本|件|天|年|岁|种|名|场|部|辆|把|句|声|"
    "倍|周|月|份|枪|发|颗|块|座|间|层|家|对|双|斤|口|头|匹|架|艘|杯|瓶|碗|步|秒|点|米|吨|"
    "集|季|代|样|首|封|片|节|根|支|台|章|招|拳|刀|下|遍|趟|回|轮|局|队|方|面|侧|边|款"
)

_UNITS: tuple[tuple[str, str], ...] = (
    ("km/h", "公里每小时"),
    ("km²", "平方公里"),
    ("m²", "平方米"),
    ("cm²", "平方厘米"),
    ("°C", "摄氏度"),
    ("°F", "华氏度"),
    ("mph", "英里每小时"),
    ("kcal", "千卡"),
    ("km", "公里"),
    ("kg", "公斤"),
    ("cm", "厘米"),
    ("mm", "毫米"),
    ("ml", "毫升"),
    ("mg", "毫克"),
    ("m", "米"),
    ("g", "克"),
    ("L", "升"),
    ("℃", "摄氏度"),
    ("℉", "华氏度"),
    ("°", "度"),
)
_UNIT_WORDS = dict(_UNITS)

_CURRENCIES = {"$": "美元", "¥": "元", "￥": "元", "€": "欧元", "£": "英镑"}

_OPERATORS = {"+": "加", "×": "乘", "÷": "除以", "=": "等于"}

_WORDS = {
    "&": "和",
    "vs.": "对",
    "vs": "对",
    "VS": "对",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT_RE = "|".join(re.escape(u) for u, _ in _UNITS)
_QTY = (
    rf"(?P<@cur>[$¥￥€£])?(?P<@num>{_NUM})(?P<@pct>%)?"
    rf"(?:\s?(?P<@unit>{_UNIT_RE})(?![A-Za-z]))?"
)
"""A number with its sign, percent and unit; `@` is replaced by a prefix for unique names."""

_PATTERNS: tuple[tuple[str, str], ...] = (
    ("date", r"(?<![\d.])(?P<y>\d{4})(?P<sep>[-/.])(?P<mo>\d{1,2})(?P=sep)(?P<d>\d{1,2})(?![\d.])"),
    ("clock", r"(?<![\d:])(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?(?![\d:])"),
    ("ratio", r"(?<![\d:])(?P<a>\d+):(?P<b>\d+)(?![\d:])"),
    ("fraction", r"(?<![\d/])(?P<fn>\d+)/(?P<fd>\d+)(?![\d/])"),
    (
        "range",
        _QTY.replace("@", "l") + r"\s?[-~～–]\s?" + _QTY.replace("@", "r") + r"(?P<rmag>[万亿]*)",
    ),
    ("neg", r"(?<![0-9A-Za-z)\]])-" + _QTY.replace("@", "n") + r"(?P<nmag>[万亿]*)"),
    ("qty", _QTY.replace("@", "q") + r"(?P<qmag>[万亿]*)"),
    ("op", r"(?<=\d)\s?(?P<sign>[+×÷=])\s?(?=[\d$¥￥€£-])"),
    ("word", r"(?<![A-Za-z])(?:Mrs\.|Mr\.|Ms\.|Dr\.|vs\.?|VS)(?![A-Za-z])|\s?&\s?"),
)
_QUANTITY = re.compile(dict(_PATTERNS)["qty"])
_MASTER = re.compile("|".join(f"(?P<{name}>{rx})" for name, rx in _PATTERNS))

_FULLWIDTH = {
    **{chr(0xFF10 + i): str(i) for i in range(10)},
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    "％": "%",
    "＋": "+",
    "＝": "=",
    "＆": "&",
    "＄": "$",
    "．": ".",
    "：": ":",
    "／": "/",
    "～": "~",
}


@dataclass(frozen=True)
class Piece:
    """`source[src_lo:src_hi]` is spoken as `text[out_lo:out_hi]`."""

    src_lo: int
    src_hi: int
    out_lo: int
    out_hi: int


@dataclass(frozen=True)
class SpeechText:
    source: str
    text: str
    """What the engine is given."""
    pieces: tuple[Piece, ...]

    def source_timings(
        self, spoken: Sequence[tuple[int, int]], duration_ms: int
    ) -> list[tuple[int, int]]:
        """`(start_ms, end_ms)` for each character of `source`, from the timings the engine
        gave for each character of `text` (empty when it gave none). A replaced run of the
        source shares the time of what was spoken for it, evenly; the result is monotonic and
        within the audio."""
        if not spoken:
            return []
        if len(spoken) != len(self.text):
            raise ValueError("spoken timings must have one entry per character of text")
        out: list[tuple[int, int]] = []
        prev = 0
        for p in self.pieces:
            n = p.src_hi - p.src_lo
            if p.out_lo < p.out_hi:
                a, b = spoken[p.out_lo][0], spoken[p.out_hi - 1][1]
            else:
                a = b = prev
            a = min(max(a, prev), duration_ms)
            b = min(max(b, a), duration_ms)
            for k in range(n):
                s = max(a + (b - a) * k // n, prev)
                e = max(a + (b - a) * (k + 1) // n, s)
                out.append((s, e))
                prev = e
        return out


def cardinal(n: int) -> str:
    """`n ≥ 0` read as an amount: 10 → 十, 1001 → 一千零一, 100000 → 十万."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return "零"
    groups: list[int] = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    if len(groups) > len(_GROUP_UNITS):
        return digits(str(sum(g * 10000**i for i, g in enumerate(groups))))
    parts: list[str] = []
    gap = False
    for i in reversed(range(len(groups))):
        g = groups[i]
        if g == 0:
            gap = bool(parts)
            continue
        if parts and (gap or g < 1000):
            parts.append("零")
        parts.append(_four(g) + _GROUP_UNITS[i])
        gap = False
    s = "".join(parts)
    return s[1:] if s.startswith("一十") else s


def _four(g: int) -> str:
    out: list[str] = []
    zero = False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = g // value % 10
        if d == 0:
            zero = bool(out)
            continue
        if zero:
            out.append("零")
            zero = False
        out.append(_DIGITS[d] + unit)
    return "".join(out)


def digits(s: str) -> str:
    """Digit by digit: "1990" → 一九九零. Non-digits are dropped."""
    return "".join(_DIGITS[int(c)] for c in s if c.isdigit())


def number(s: str) -> str:
    """A written number (thousands commas and a decimal part allowed) read as an amount."""
    whole, _, frac = s.replace(",", "").partition(".")
    out = cardinal(int(whole))
    return out + "点" + digits(frac) if frac else out


def normalize(text: str) -> SpeechText:
    """The text an engine should be given for `text`, with the mapping back."""
    folded = "".join(_FULLWIDTH.get(c, c) for c in text)
    b = _Builder(text)
    pos = 0
    for m in _MASTER.finditer(folded):
        spoken = _spoken(m, folded)
        # not what it looked like (a 13th month, 1/0): each number on its own
        found = (
            [(m, spoken)]
            if spoken is not None
            else [(q, _qty(q, folded)) for q in _QUANTITY.finditer(folded, m.start(), m.end())]
        )
        for q, said in found:
            assert said is not None
            b.keep(pos, q.start())
            b.replace(q.start(), q.end(), said)
            pos = q.end()
    b.keep(pos, len(text))
    return b.build()


class _Builder:
    def __init__(self, source: str) -> None:
        self.source = source
        self.out: list[str] = []
        self.n_out = 0
        self.pieces: list[Piece] = []

    def keep(self, lo: int, hi: int) -> None:
        for k in range(lo, hi):
            self._emit(k, k + 1, self.source[k])

    def replace(self, lo: int, hi: int, spoken: str) -> None:
        self._emit(lo, hi, spoken)

    def _emit(self, lo: int, hi: int, s: str) -> None:
        self.pieces.append(Piece(lo, hi, self.n_out, self.n_out + len(s)))
        self.out.append(s)
        self.n_out += len(s)

    def build(self) -> SpeechText:
        return SpeechText(self.source, "".join(self.out), tuple(self.pieces))


def _spoken(m: re.Match[str], text: str) -> str | None:
    kind = m.lastgroup
    assert kind is not None
    return _HANDLERS[kind](m, text)


def _date(m: re.Match[str], text: str) -> str | None:
    mo, d = int(m["mo"]), int(m["d"])
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{digits(m['y'])}年{cardinal(mo)}月{cardinal(d)}日"


def _clock(m: re.Match[str], text: str) -> str | None:
    h, mi = int(m["h"]), int(m["mi"])
    s = int(m["s"]) if m["s"] else 0
    if h > 24 or mi > 59 or s > 59:
        return _ratio_of(m["h"], m["mi"]) if m["s"] is None else None
    out = ("两" if h == 2 else cardinal(h)) + "点"
    if mi or s:
        out += ("零" if mi < 10 else "") + cardinal(mi) + "分"
    if s:
        out += ("零" if s < 10 else "") + cardinal(s) + "秒"
    return out


def _ratio(m: re.Match[str], text: str) -> str | None:
    return _ratio_of(m["a"], m["b"])


def _ratio_of(a: str, b: str) -> str:
    return number(a) + "比" + number(b)


def _fraction(m: re.Match[str], text: str) -> str | None:
    if int(m["fd"]) == 0:
        return None
    return number(m["fd"]) + "分之" + number(m["fn"])


def _range(m: re.Match[str], text: str) -> str | None:
    left, right = m["lnum"], m["rnum"]
    after = text[m.end() :]
    if _is_year(right, after) and _is_year(left, "年"):
        return digits(left) + "到" + digits(right)
    bare = not any(m[g] for g in ("lcur", "lpct", "lunit", "rcur", "rpct", "runit", "rmag"))
    if bare and left.isdigit() and right.isdigit() and int(left) >= int(right):
        return number(left) + "比" + number(right)  # a score
    # a unit, sign or magnitude written once covers both ends: 3-5km, 30-50%, $1-2万
    mag = m["rmag"]
    lo = _amount(left, text, m.start("lnum"), after=mag + after) + mag
    hi = _amount(right, text, m.start("rnum"), after=mag + after) + mag
    if m["rpct"] or m["lpct"]:
        lo, hi = "百分之" + lo, "百分之" + hi
    unit, cur = m["runit"] or m["lunit"], m["rcur"] or m["lcur"]
    return lo + "到" + hi + (_UNIT_WORDS[unit] if unit else "") + (_CURRENCIES[cur] if cur else "")


def _neg(m: re.Match[str], text: str) -> str | None:
    return "负" + _quantity(m, text, "n")


def _qty(m: re.Match[str], text: str) -> str | None:
    return _quantity(m, text, "q")


def _quantity(m: re.Match[str], text: str, p: str) -> str:
    num, cur, unit, mag = m[p + "num"], m[p + "cur"], m[p + "unit"], m[p + "mag"]
    after = text[m.end() :]
    if cur is None and unit is None and not m[p + "pct"] and not mag and _is_year(num, after):
        return digits(num)
    out = _amount(num, text, m.start(p + "num"), after=mag or after)
    if m[p + "pct"]:
        out = "百分之" + out
    return out + mag + (_UNIT_WORDS[unit] if unit else "") + (_CURRENCIES[cur] if cur else "")


def _amount(num: str, text: str, start: int, *, after: str) -> str:
    """`num` (starting at `text[start]`) read as an amount, digit by digit, or with 两."""
    before = text[:start]
    if before[-1:].isascii() and before[-1:].isalpha():
        return digits(num)  # B612, MP3
    plain = num.isdigit()
    if plain and ((len(num) > 1 and num[0] == "0") or len(num) >= 11):
        return digits(num)
    out = number(num)
    if plain and not before.endswith("第"):
        if num == "2" and re.match(rf"\s?(?:{_MEASURE_WORDS}|[千万亿])", after):
            return "两"
        if out[0] == "二" and out[1:2] in ("千", "万", "亿"):
            return "两" + out[1:]
    return out


def _is_year(num: str, after: str) -> bool:
    return len(num) == 4 and num.isdigit() and after.lstrip()[:1] == "年"


def _op(m: re.Match[str], text: str) -> str | None:
    return _OPERATORS[m["sign"]]


def _word(m: re.Match[str], text: str) -> str | None:
    return _WORDS[m.group().strip()]


_HANDLERS: dict[str, Callable[[re.Match[str], str], str | None]] = {
    "date": _date,
    "clock": _clock,
    "ratio": _ratio,
    "fraction": _fraction,
    "range": _range,
    "neg": _neg,
    "qty": _qty,
    "op": _op,
    "word": _word,
}
