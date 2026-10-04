"""Shared value types for all data layers. Pure: stdlib + pydantic only."""

from __future__ import annotations

import os
import re
import time
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = r"[0-9A-HJKMNP-TV-Z]{26}"

ID_PREFIXES = ("ast", "prj", "ln", "sh", "sc", "ch", "seg", "job", "scr", "pln", "rnd", "spk")


def new_ulid(now_ms: int | None = None, rand: bytes | None = None) -> str:
    """26-char Crockford-base32 ULID: 48-bit ms timestamp + 80-bit randomness."""
    ts = int(time.time() * 1000) if now_ms is None else now_ms
    if not 0 <= ts < 1 << 48:
        raise ValueError("timestamp out of ULID range")
    r = int.from_bytes(os.urandom(10) if rand is None else rand, "big")
    value = (ts << 80) | r
    return "".join(_CROCKFORD[(value >> (5 * i)) & 31] for i in range(25, -1, -1))


def new_id(prefix: str) -> str:
    if prefix not in ID_PREFIXES:
        raise ValueError(f"unknown id prefix: {prefix}")
    return f"{prefix}_{new_ulid()}"


# Fixtures and hand-written ids may use short readable suffixes, so ULIDs are not
# enforced here; the prefix is what prevents mixing up kinds of ids.
AssetId = Annotated[str, StringConstraints(pattern=r"^ast_[0-9A-Za-z]+$")]
ProjectId = Annotated[str, StringConstraints(pattern=r"^prj_[0-9A-Za-z]+$")]
LineId = Annotated[str, StringConstraints(pattern=r"^ln_[0-9A-Za-z]+$")]
ShotId = Annotated[str, StringConstraints(pattern=r"^sh_[0-9A-Za-z]+$")]
SceneId = Annotated[str, StringConstraints(pattern=r"^sc_[0-9A-Za-z]+$")]
CharacterId = Annotated[str, StringConstraints(pattern=r"^ch_[0-9A-Za-z]+$")]
SegmentId = Annotated[str, StringConstraints(pattern=r"^seg_[0-9A-Za-z]+$")]
JobId = Annotated[str, StringConstraints(pattern=r"^job_[0-9A-Za-z]+$")]
ScriptId = Annotated[str, StringConstraints(pattern=r"^scr_[0-9A-Za-z]+$")]
PlanId = Annotated[str, StringConstraints(pattern=r"^pln_[0-9A-Za-z]+$")]
RenderId = Annotated[str, StringConstraints(pattern=r"^rnd_[0-9A-Za-z]+$")]

TimeMs = Annotated[int, Field(ge=0)]
"""Source time in integer milliseconds. Never floats (ARCHITECTURE §5.1)."""

Frame = Annotated[int, Field(ge=0)]
"""Output timeline position as an integer frame index."""

Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{8,64}$")]


class Strict(BaseModel):
    """Base for all domain models: immutable, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Rational(Strict):
    """Exact fraction, used for frame rates (23.976 -> 24000/1001)."""

    num: int = Field(gt=0)
    den: int = Field(gt=0)

    def as_float(self) -> float:
        return self.num / self.den

    def frames_for_ms(self, ms: int) -> int:
        """Whole frames covering `ms`, rounded to nearest."""
        return (ms * self.num + 500 * self.den) // (1000 * self.den)

    def ms_for_frames(self, frames: int) -> int:
        """Milliseconds of `frames`, rounded to nearest."""
        return (frames * self.den * 1000 + self.num // 2) // self.num


class TimeRange(Strict):
    """Half-open range [start_ms, end_ms) in source time."""

    start_ms: TimeMs
    end_ms: TimeMs

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start_ms >= self.end_ms:
            raise ValueError(f"start_ms ({self.start_ms}) must be < end_ms ({self.end_ms})")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def overlaps(self, other: TimeRange) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

    def contains(self, ms: int) -> bool:
        return self.start_ms <= ms < self.end_ms


class Versioned(Strict):
    """Base for persisted documents. Bump SCHEMA_VERSION on breaking changes and
    add a migration function (ARCHITECTURE §5.1)."""

    SCHEMA_VERSION: ClassVar[int] = 1
    schema_version: int = 1

    @model_validator(mode="after")
    def _known_version(self) -> Self:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError(
                f"{type(self).__name__}: schema_version {self.schema_version} "
                f"!= supported {self.SCHEMA_VERSION}; run migration"
            )
        return self


def canonical_json(model: BaseModel) -> str:
    """Deterministic serialization: sorted keys, UTF-8 (no ascii escaping), trailing
    newline. Same content -> same bytes -> stable content hashes."""
    import json

    data = model.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def id_prefix(value: str) -> str:
    m = re.match(r"^([a-z]+)_", value)
    if not m:
        raise ValueError(f"not a prefixed id: {value!r}")
    return m.group(1)
