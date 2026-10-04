import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from offscreen.domain.common import canonical_json
from offscreen.domain.index import Scene, Shots
from offscreen.domain.job import can_transition
from offscreen.domain.plan import EditPlan, PlanSegment
from offscreen.domain.registry import DOCUMENTS
from offscreen.domain.schemas import export_schemas
from offscreen.domain.script import Script, ScriptSegment
from offscreen.domain.timeline import Timeline

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "domain"
REPO = Path(__file__).resolve().parents[3]


def load(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_every_document_type_has_a_fixture() -> None:
    assert {p.stem for p in FIXTURES.glob("*.json")} == set(DOCUMENTS)


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_fixture_roundtrip(name: str) -> None:
    cls = DOCUMENTS[name]
    model = cls.model_validate(load(name))
    again = cls.model_validate_json(canonical_json(model))
    assert again == model
    assert canonical_json(again) == canonical_json(model)


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_unknown_fields_rejected(name: str) -> None:
    data = load(name)
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        DOCUMENTS[name].model_validate(data)


def test_committed_json_schemas_are_up_to_date() -> None:
    out = REPO / "docs" / "schemas"
    for name, text in export_schemas().items():
        f = out / f"{name}.schema.json"
        assert f.exists(), f"missing {f}; run `make schemas`"
        assert f.read_text(encoding="utf-8") == text, f"{name} schema drifted; run `make schemas`"


def test_shots_must_not_overlap() -> None:
    data = load("shots")
    data["shots"][1]["start_ms"] = 1233000
    with pytest.raises(ValidationError, match="overlap"):
        Shots.model_validate(data)


def test_time_ranges_must_be_ordered() -> None:
    data = load("scenes")["scenes"][0]
    data["end_ms"] = data["start_ms"]
    with pytest.raises(ValidationError):
        Scene.model_validate(data)


def test_narration_segment_needs_scene_refs() -> None:
    with pytest.raises(ValidationError, match="scene_refs"):
        ScriptSegment(id="seg_x", kind="narration", text="hello")
    with pytest.raises(ValidationError, match="line_refs"):
        ScriptSegment(id="seg_x", kind="original", text="（原声）")


def test_script_rejects_dangling_annotation_and_bad_parent() -> None:
    data = load("script")
    data["annotations"][0]["segment_id"] = "seg_99"
    with pytest.raises(ValidationError, match="unknown segments"):
        Script.model_validate(data)
    data = load("script")
    data["parent_version"] = data["version"]
    with pytest.raises(ValidationError, match="parent_version"):
        Script.model_validate(data)


def test_plan_stale_narration_may_lack_audio_but_fresh_may_not() -> None:
    seg = load("plan")["segments"][0]
    seg.update(stale=True, audio=None, clips=[])
    PlanSegment.model_validate(seg)
    seg.update(stale=False)
    with pytest.raises(ValidationError, match="audio and clips"):
        PlanSegment.model_validate(seg)


def test_plan_original_segment_has_no_audio() -> None:
    seg = load("plan")["segments"][1]
    seg["audio"] = load("plan")["segments"][0]["audio"]
    with pytest.raises(ValidationError, match="no narration audio"):
        PlanSegment.model_validate(seg)
    EditPlan.model_validate(load("plan"))


def test_timeline_video_track_must_tile_exactly() -> None:
    data = load("timeline")
    data["video"][1]["f0"] = 160  # gap
    with pytest.raises(ValidationError, match="gap/overlap"):
        Timeline.model_validate(data)
    data = load("timeline")
    data["duration_frames"] = 260  # video ends early
    with pytest.raises(ValidationError, match="duration_frames"):
        Timeline.model_validate(data)
    data = load("timeline")
    data["subtitles"][0]["f1"] = 999
    with pytest.raises(ValidationError, match="past duration"):
        Timeline.model_validate(data)


def test_job_state_machine() -> None:
    assert can_transition("queued", "running")
    assert can_transition("running", "queued")  # crash recovery
    assert can_transition("failed", "queued")  # manual retry
    assert not can_transition("succeeded", "running")
    assert not can_transition("canceled", "queued")


def test_plan_narration_text_snapshot_rules() -> None:
    seg = load("plan")["segments"][0]
    PlanSegment.model_validate(seg)

    stale = {**seg, "stale": True, "audio": None, "clips": [], "text": None}
    PlanSegment.model_validate(stale)  # a segment waiting for re-synthesis needs no text

    with pytest.raises(ValidationError, match="needs its text"):
        PlanSegment.model_validate({**seg, "text": None})
    with pytest.raises(ValidationError, match="needs its text"):
        PlanSegment.model_validate({**seg, "text": "  ", "text_hash": None})
    with pytest.raises(ValidationError, match="does not match"):
        PlanSegment.model_validate({**seg, "text": "另一段话"})
