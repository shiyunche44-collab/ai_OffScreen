"""Parse subtitle text (SRT, ASS/SSA) into timed cues. Pure: callers do the file IO."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


class SubtitleParseError(ValueError):
    pass


_SRT_TIME = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
_ASS_TIME = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})[.:](\d{1,3})$")
_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_ASS_DRAWING = re.compile(r"\{[^}]*\\p[1-9][^}]*\}")


def _ms(h: str, m: str, s: str, frac: str) -> int:
    # SRT has milliseconds, ASS centiseconds; both are fractions of a second, so right-pad.
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(frac.ljust(3, "0")[:3])


def _clean(text: str) -> str:
    text = _ASS_OVERRIDE.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"\\[Nn]", " ", text).replace("\\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = _SRT_TIME.search(line)
            if m:
                g = m.groups()
                body = _clean(" ".join(lines[i + 1 :]))
                start, end = _ms(*g[:4]), _ms(*g[4:])
                if body and end > start:
                    cues.append(Cue(start, end, body))
                break
    return cues


def parse_ass(text: str) -> list[Cue]:
    fields: list[str] | None = None
    cues: list[Cue] = []
    in_events = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
        elif not in_events:
            continue
        elif line.lower().startswith("format:"):
            fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
        elif line.lower().startswith("dialogue:"):
            if fields is None:
                raise SubtitleParseError("ASS: Dialogue before Format line")
            parts = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
            row = dict(zip(fields, parts, strict=False))
            if "start" not in row or "end" not in row or "text" not in row:
                raise SubtitleParseError("ASS: Format lacks Start/End/Text")
            ms = []
            for key in ("start", "end"):
                m = _ASS_TIME.match(row[key].strip())
                if not m:
                    raise SubtitleParseError(f"ASS: bad time {row[key]!r}")
                h, mi, s, frac = m.groups()
                ms.append(_ms(h, mi, s, frac))
            if _ASS_DRAWING.search(row["text"]):
                continue  # vector drawings (logos, karaoke effects), not speech
            body = _clean(row["text"])
            if body and ms[1] > ms[0]:
                cues.append(Cue(ms[0], ms[1], body))
    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    return cues


def parse_subtitles(text: str, ext: str) -> list[Cue]:
    ext = ext.lower()
    if ext == ".srt":
        return parse_srt(text)
    if ext in (".ass", ".ssa"):
        return parse_ass(text)
    raise SubtitleParseError(f"unsupported subtitle format {ext!r}")


def guess_language(cues: list[Cue]) -> str:
    """Coarse script-based guess: "zh", "ja", "ko", else "und" (undetermined)."""
    text = "".join(c.text for c in cues)
    kana = sum("\u3040" <= ch <= "\u30ff" for ch in text)
    hangul = sum("\uac00" <= ch <= "\ud7af" for ch in text)
    han = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    if not letters:
        return "und"
    if kana / letters > 0.05:
        return "ja"
    if hangul / letters > 0.2:
        return "ko"
    if han / letters > 0.2:
        return "zh"
    return "und"
