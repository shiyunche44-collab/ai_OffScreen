import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from offscreen.domain.common import (
    Rational,
    TimeRange,
    Versioned,
    canonical_json,
    new_id,
    new_ulid,
)
from offscreen.store.files import atomic_write_bytes, read_model, write_model


def test_ulid_shape_and_ordering() -> None:
    a = new_ulid(now_ms=1_000, rand=b"\x00" * 10)
    b = new_ulid(now_ms=2_000, rand=b"\x00" * 10)
    assert len(a) == 26 and a < b


def test_new_id_prefix() -> None:
    assert new_id("seg").startswith("seg_")
    with pytest.raises(ValueError):
        new_id("nope")


def test_time_range_rules() -> None:
    assert TimeRange(start_ms=0, end_ms=10).duration_ms == 10
    with pytest.raises(ValidationError):
        TimeRange(start_ms=10, end_ms=10)
    with pytest.raises(ValidationError):
        TimeRange(start_ms=-1, end_ms=5)
    assert not TimeRange(start_ms=0, end_ms=10).overlaps(TimeRange(start_ms=10, end_ms=20))
    assert TimeRange(start_ms=0, end_ms=11).overlaps(TimeRange(start_ms=10, end_ms=20))


def test_rational_rejects_float_and_zero() -> None:
    assert Rational(num=24000, den=1001).as_float() == pytest.approx(23.976, abs=1e-3)
    with pytest.raises(ValidationError):
        Rational(num=24, den=0)


@given(
    st.integers(min_value=0, max_value=10_000_000),
    st.sampled_from([(24, 1), (30, 1), (24000, 1001)]),
)
def test_frame_ms_roundtrip_within_one_frame(ms: int, fps: tuple[int, int]) -> None:
    r = Rational(num=fps[0], den=fps[1])
    frame_ms = 1000 * r.den / r.num
    assert abs(r.ms_for_frames(r.frames_for_ms(ms)) - ms) <= frame_ms / 2 + 1


class Doc(Versioned):
    name: str


def test_versioned_rejects_unknown_schema() -> None:
    Doc(name="x")
    with pytest.raises(ValidationError):
        Doc(name="x", schema_version=99)
    with pytest.raises(ValidationError):
        Doc.model_validate({"name": "x", "extra": 1})


def test_canonical_json_is_deterministic() -> None:
    class M(BaseModel):
        b: int
        a: str

    assert canonical_json(M(b=1, a="电影")) == canonical_json(M(a="电影", b=1))
    assert "电影" in canonical_json(M(b=1, a="电影"))  # not ascii-escaped


def test_atomic_write_and_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "a" / "doc.json"
    h1 = write_model(p, Doc(name="电影"))
    assert read_model(p, Doc) == Doc(name="电影")
    assert write_model(p, Doc(name="电影")) == h1
    assert not list(p.parent.glob("*.tmp"))


def test_atomic_write_cleans_up_on_failure(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os

    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        atomic_write_bytes(tmp_path / "x.json", b"{}")
    assert not list(tmp_path.glob("*"))
