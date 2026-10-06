"""A writing style preset: the knobs that make one commentary sound different from another.

Presets are plain YAML files (ARCHITECTURE §7.2), not code. The writing steps (M4-04 / M4-05)
render them into their prompts and the rule checker (M4-06) enforces `banned_words`."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import Strict

STYLE_ID_RE = r"^[a-z][a-z0-9_]{0,31}$"  # pydantic patterns search, so anchor them
SHARE_TOLERANCE = 0.01


class StyleBeat(Strict):
    """One step of the style's structure template."""

    name: str = Field(pattern=STYLE_ID_RE)
    """Stable label the model puts on a segment (`beat`), e.g. `hook`, `climax`."""
    purpose: str = Field(min_length=1)
    """What this part does, in the words given to the model."""
    share: float = Field(gt=0, le=1)
    """Rough fraction of the whole text; all shares add up to 1."""


class StylePreset(Strict):
    id: str = Field(pattern=STYLE_ID_RE)
    name: str = Field(min_length=1)
    """Display name."""
    description: str = Field(min_length=1)
    """One line for the picker: who it suits."""
    tone: str = Field(min_length=1)
    """How it sounds: voice, attitude, rhythm."""
    perspective: Literal["first", "third"] = "third"
    """Person the narration is told in."""
    structure: list[StyleBeat] = Field(min_length=1)
    hook_types: list[str] = Field(min_length=1)
    """Ways the first segment may open, as instructions ("以一个悬而未决的问题开场")."""
    phrases: list[str] = []
    """Typical sentence patterns, shown to the model as examples of the voice."""
    banned_words: list[str] = []
    """Words and clichés the text must not contain."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [b.name for b in self.structure]
        if len(set(names)) != len(names):
            raise ValueError("beat names must be unique")
        total = sum(b.share for b in self.structure)
        if abs(total - 1.0) > SHARE_TOLERANCE:
            raise ValueError(f"beat shares must add up to 1, got {total:.3f}")
        for field in ("hook_types", "phrases", "banned_words"):
            items = getattr(self, field)
            if any(not s.strip() for s in items):
                raise ValueError(f"{field} must not contain empty entries")
            if len(set(items)) != len(items):
                raise ValueError(f"{field} must not repeat an entry")
        hits = [w for w in self.banned_words if any(w in p for p in self.phrases)]
        if hits:
            raise ValueError(f"phrases use banned words: {', '.join(hits)}")
        return self

    def banned_in(self, text: str) -> list[str]:
        """The banned words that occur in `text`, in the order the preset lists them."""
        return [w for w in self.banned_words if w in text]
