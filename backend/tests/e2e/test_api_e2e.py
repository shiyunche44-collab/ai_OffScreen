"""The HTTP API over real services (fake LLM / TTS / shot detection, real ffmpeg and stores).
A Worker runs the queued jobs, as the server process will."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.domain.plan import EditPlan
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.services.app import AppServices
from offscreen.services.pipeline import Providers
from offscreen.worker import Worker, WorkerSettings

FAST = WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=0.0)


@pytest.fixture
def services(cfg: AppConfig, fakes: Providers, movie: Path) -> Iterator[AppServices]:
    rooted = cfg.model_copy(update={"media_roots": [movie.parent]})
    with AppServices(rooted, providers=fakes) as s:
        yield s


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def drain(services: AppServices) -> None:
    Worker(
        services.db and services.jobs.jobs, services.jobs.execute, services.cfg.data_dir, FAST
    ).drain(timeout_s=120)


def err(r: Any, status: int, code: str) -> dict[str, Any]:
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code, body
    assert body["error"]["message"]
    return body["error"]  # type: ignore[no-any-return]


def import_movie(client: TestClient, movie: Path) -> dict[str, Any]:
    r = client.post("/api/assets", json={"path": str(movie)})
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


# --- assets -----------------------------------------------------------------------------


def test_import_list_and_get_an_asset(client: TestClient, movie: Path) -> None:
    asset = import_movie(client, movie)
    assert asset["id"].startswith("ast_") and asset["title"]
    again = import_movie(client, movie)
    assert again["id"] == asset["id"]  # same file, same asset
    listed = client.get("/api/assets").json()
    assert [a["asset"]["id"] for a in listed] == [asset["id"]]
    assert not any(s["cached"] for s in listed[0]["stages"])  # status comes with the list

    detail = client.get(f"/api/assets/{asset['id']}").json()
    assert detail["asset"]["id"] == asset["id"]
    assert [s["stage"] for s in detail["stages"]] == [
        "analysis.proxy",
        "analysis.shots",
        "analysis.keyframes",
        "analysis.transcript",
        "analysis.captions",
        "analysis.scenes",
        "analysis.story",
    ]
    assert not any(s["cached"] for s in detail["stages"])


# --- browsing the media roots -----------------------------------------------------------


def test_browse_lists_the_roots_then_folders_and_video_files(
    client: TestClient, movie: Path, tmp_path: Path
) -> None:
    root = movie.parent
    (root / "Extras").mkdir()
    (root / "Extras" / "clip.MKV").write_bytes(b"x")
    (root / "Aardvark").mkdir()
    (root / "notes.txt").write_text("not a video")
    (root / ".hidden.mp4").write_bytes(b"x")
    (root / "Sample.srt").write_text("subtitles are not listed")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.mp4").write_bytes(b"x")
    (root / "escape").symlink_to(outside)  # a link out of the roots
    (root / "leak.mp4").symlink_to(outside / "leak.mp4")

    top = client.get("/api/assets/browse").json()
    assert top["path"] is None and top["parent"] is None
    assert [e["path"] for e in top["entries"]] == [str(root.resolve())]
    assert top["entries"][0]["kind"] == "dir"

    inside = client.get("/api/assets/browse", params={"path": str(root)}).json()
    assert inside["path"] == str(root.resolve())
    assert inside["parent"] is None  # up from a root is the top level
    assert [(e["name"], e["kind"]) for e in inside["entries"]] == [
        ("Aardvark", "dir"),
        ("Extras", "dir"),
        ("Sample.mp4", "file"),  # dirs first, then files; no hidden, non-video or escaping links
    ]
    sample = inside["entries"][2]
    assert sample["size"] == movie.stat().st_size and sample["asset_id"] is None

    sub = client.get("/api/assets/browse", params={"path": str(root / "Extras")}).json()
    assert sub["parent"] == str(root.resolve())
    assert [e["name"] for e in sub["entries"]] == ["clip.MKV"]


def test_browse_marks_files_that_are_already_imported(client: TestClient, movie: Path) -> None:
    asset = import_movie(client, movie)
    listing = client.get("/api/assets/browse", params={"path": str(movie.parent)}).json()
    (entry,) = [e for e in listing["entries"] if e["name"] == movie.name]
    assert entry["asset_id"] == asset["id"]


def test_browse_cannot_leave_the_media_roots(
    client: TestClient, movie: Path, tmp_path: Path
) -> None:
    for path in (str(tmp_path), str(movie.parent / ".."), "/", "/etc"):
        err(client.get("/api/assets/browse", params={"path": path}), 422, "invalid_input")
    err(client.get("/api/assets/browse", params={"path": str(movie)}), 404, "not_found")  # a file
    err(
        client.get("/api/assets/browse", params={"path": str(movie.parent / "nope")}),
        404,
        "not_found",
    )


def test_browse_without_media_roots_says_what_to_configure(
    cfg: AppConfig, fakes: Providers
) -> None:
    with AppServices(cfg, providers=fakes) as s, TestClient(create_app(s)) as c:
        e = err(c.get("/api/assets/browse"), 422, "invalid_input")
        assert "media_roots" in e["message"]


def test_import_is_limited_to_the_media_roots(
    client: TestClient, movie: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(movie.read_bytes())
    err(client.post("/api/assets", json={"path": str(outside)}), 422, "invalid_input")
    sneaky = movie.parent / ".." / "elsewhere.mp4"
    err(client.post("/api/assets", json={"path": str(sneaky)}), 422, "invalid_input")
    err(
        client.post("/api/assets", json={"path": str(movie.parent)}), 422, "invalid_input"
    )  # a folder
    err(
        client.post("/api/assets", json={"path": str(movie.parent / "nope.mp4")}),
        422,
        "invalid_input",
    )
    assert client.get("/api/assets").json() == []


def test_a_file_that_is_not_a_video_is_refused(client: TestClient, movie: Path) -> None:
    junk = movie.parent / "notes.mp4"
    junk.write_text("this is not a movie")
    err(client.post("/api/assets", json={"path": str(junk)}), 422, "invalid_input")


def test_import_without_media_roots_explains_what_to_configure(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    with AppServices(cfg, providers=fakes) as s, TestClient(create_app(s)) as c:
        e = err(c.post("/api/assets", json={"path": str(movie)}), 422, "invalid_input")
        assert "media_roots" in e["message"]


def test_unknown_asset_is_a_404_with_the_error_shape(client: TestClient) -> None:
    err(client.get("/api/assets/ast_missing"), 404, "not_found")
    err(client.post("/api/assets/ast_missing/analyze"), 404, "not_found")


def test_request_validation_errors_list_the_fields(client: TestClient) -> None:
    e = err(client.post("/api/assets", json={"title": "x"}), 422, "validation_error")
    assert e["details"] and e["details"][0]["loc"] == ["body", "path"]
    err(client.post("/api/assets", json={"path": ""}), 422, "validation_error")


def test_analyze_runs_as_a_job_and_updates_the_asset_status(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    r = client.post(f"/api/assets/{asset['id']}/analyze")
    assert r.status_code == 202
    job = r.json()
    assert (job["status"], job["stage"], job["lane"]) == ("queued", "analysis.story", "api")
    assert (
        client.post(f"/api/assets/{asset['id']}/analyze").json()["id"] == job["id"]
    )  # no duplicate

    drain(services)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert (done["status"], done["progress"]) == ("succeeded", 1.0)
    log = client.get(f"/api/jobs/{job['id']}/log")
    assert log.status_code == 200 and log.headers["content-type"].startswith("text/plain")
    assert "start analysis.story" in log.text

    stages = client.get(f"/api/assets/{asset['id']}").json()["stages"]
    assert all(s["cached"] for s in stages)


# --- projects ---------------------------------------------------------------------------


def test_create_project_defaults_and_validation(client: TestClient, movie: Path) -> None:
    asset = import_movie(client, movie)
    r = client.post("/api/projects", json={"asset_id": asset["id"]})
    assert r.status_code == 201
    project = r.json()
    assert project["id"].startswith("prj_") and project["name"] == asset["title"]
    assert project["options"] == {
        "minutes": 3.0,
        "voice": None,
        "style": "suspense",
        "spoil_ending": True,
    }

    named = client.post(
        "/api/projects",
        json={
            "asset_id": asset["id"],
            "name": "短版",
            "options": {"minutes": 0.25, "style": "roast"},
        },
    ).json()
    assert named["name"] == "短版" and named["options"]["minutes"] == 0.25
    assert [p["id"] for p in client.get("/api/projects").json()] == [named["id"], project["id"]]

    err(client.post("/api/projects", json={"asset_id": "ast_missing"}), 404, "not_found")
    err(client.post("/api/projects", json={"asset_id": "nonsense"}), 422, "validation_error")
    bad = {"asset_id": asset["id"], "options": {"minutes": 0}}
    err(client.post("/api/projects", json=bad), 422, "validation_error")
    unknown_style = {"asset_id": asset["id"], "options": {"style": "funny"}}
    err(client.post("/api/projects", json=unknown_style), 422, "invalid_input")
    err(client.get("/api/projects/prj_missing"), 404, "not_found")
    err(client.get("/api/projects/prj_missing/script"), 404, "not_found")


def test_styles_are_listed_and_project_options_can_be_changed(
    client: TestClient, movie: Path
) -> None:
    styles = client.get("/api/styles").json()
    assert [s["id"] for s in styles] == ["emotional", "roast", "suspense"]
    assert {"name", "description", "tone", "structure"} <= set(styles[0])

    asset = import_movie(client, movie)
    pid = client.post("/api/projects", json={"asset_id": asset["id"]}).json()["id"]
    url = f"/api/projects/{pid}"
    changed = client.patch(
        url,
        json={"name": "新名字", "options": {"minutes": 2, "style": "roast", "spoil_ending": False}},
    )
    assert changed.status_code == 200
    assert changed.json()["name"] == "新名字"
    assert changed.json()["options"] == {
        "minutes": 2.0,
        "voice": None,
        "style": "roast",
        "spoil_ending": False,
    }
    assert client.get(url).json()["project"]["options"]["style"] == "roast"

    only_name = client.patch(url, json={"name": "  再改  "}).json()
    assert only_name["name"] == "再改" and only_name["options"]["style"] == "roast"  # kept
    assert client.patch(url, json={}).json()["name"] == "再改"

    err(client.patch(url, json={"options": {"style": "funny"}}), 422, "invalid_input")
    err(client.patch(url, json={"name": " "}), 422, "invalid_input")
    err(client.patch(url, json={"options": {"minutes": 0}}), 422, "validation_error")
    err(client.patch("/api/projects/prj_missing", json={"name": "x"}), 404, "not_found")
    assert client.get(url).json()["project"]["options"]["style"] == "roast"  # nothing half-saved


def test_a_project_goes_from_nothing_to_a_playable_video(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    project = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()
    pid = project["id"]
    before = client.get(f"/api/projects/{pid}").json()
    assert before["video"] is None and not any(s["cached"] for s in before["stages"])
    err(client.get(f"/api/projects/{pid}/script"), 404, "not_found")  # nothing generated yet

    jobs = []
    for step in ("script:generate", "plan:build", "render"):
        # one after the other, as the page does: the plan follows the script that was stored
        r = client.post(f"/api/projects/{pid}/{step}")
        assert r.status_code == 202, r.text
        jobs.append(r.json())
        drain(services)
    assert [j["stage"] for j in jobs] == ["creation.script", "creation.plan", "output.render"]
    assert all(client.get(f"/api/jobs/{j['id']}").json()["status"] == "succeeded" for j in jobs)

    script = client.get(f"/api/projects/{pid}/script").json()
    assert script["params"]["target_duration_s"] == 15
    assert [seg["kind"] for seg in script["segments"]] == ["narration"] * 3
    assert all(seg["text"] for seg in script["segments"])
    # the generated draft is the project's first script version, with the project's own ids
    assert (script["version"], script["author"], script["project_id"]) == (1, "ai", pid)
    assert [v["version"] for v in client.get(f"/api/projects/{pid}/script/versions").json()] == [1]
    # the fact check ran in the same job and its findings are part of that version
    assert [(a["segment_id"], a["type"]) for a in script["annotations"]] == [
        ("seg_01", "fact_check")
    ]

    after = client.get(f"/api/projects/{pid}").json()
    assert [s["stage"] for s in after["stages"] if not s["cached"]] == []
    assert after["video"].endswith("/final.mp4")
    assert (services.cfg.data_dir / after["video"]).is_file()

    other = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.3}}
    ).json()
    cached = {
        s["stage"]
        for s in client.get(f"/api/projects/{other['id']}").json()["stages"]
        if s["cached"]
    }
    assert cached == {  # same film, other length: only the analysis carries over
        "analysis.proxy",
        "analysis.shots",
        "analysis.keyframes",
        "analysis.transcript",
        "analysis.captions",
        "analysis.scenes",
        "analysis.story",
    }
    assert client.get(f"/api/projects/{other['id']}").json()["video"] is None


def test_the_plan_follows_the_edited_script_and_redoes_only_what_changed(
    client: TestClient, services: AppServices, fakes: Providers, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    script_url = f"/api/projects/{pid}/script"

    def run(step: str) -> dict[str, Any]:
        job = client.post(f"/api/projects/{pid}/{step}").json()
        drain(services)
        done = client.get(f"/api/jobs/{job['id']}").json()
        assert done["status"] == "succeeded", done
        return done  # type: ignore[no-any-return]

    def head() -> EditPlan:
        plan = services.jobs.docs.read(pid, "plan", EditPlan)
        assert plan is not None
        return plan

    assert isinstance(fakes.tts, FakeTTS)
    tts = fakes.tts
    run("script:generate")
    run("plan:build")
    v1 = head()
    spoken = len(tts.calls)
    assert (v1.version, v1.parent_version, v1.author) == (1, None, "ai")
    assert v1.id == f"pln_{pid.split('_', 1)[1]}" and v1.project_id == pid
    assert [s.id for s in v1.segments] == ["seg_01", "seg_02", "seg_03"] and spoken == 3

    # building again changes nothing: no new version, nobody is asked to speak again
    run("plan:build")
    assert head() == v1 and len(tts.calls) == spoken

    # a person rewrites the first segment; the plan redoes that segment alone
    script = client.get(script_url).json()
    edit = {k: script[k] for k in ("params", "outline", "segments", "annotations")}
    edit["segments"][0]["text"] = "我改过的开场白，比原来长一些。"
    client.put(script_url, json={**edit, "base_version": script["version"]})
    job = run("plan:build")
    v2 = head()
    assert (v2.version, v2.parent_version, v2.id) == (2, 1, v1.id)
    assert v2.script_ref.version == script["version"] + 1
    assert [c[0] for c in tts.calls[spoken:]] == ["我改过的开场白，比原来长一些。"]
    assert v2.segments[0].text == "我改过的开场白，比原来长一些。"
    assert v2.segments[1:] == v1.segments[1:]
    assert "plan: rebuilt seg_01 (text)" in services.jobs.log(job["id"])
    assert "plan: reused seg_02" in services.jobs.log(job["id"])
    # every version's audio is on disk next to the versions
    audio_dir = services.jobs.docs.dir_for(pid, "plan")
    for plan in (v1, v2):
        assert all((audio_dir / s.audio.file).is_file() for s in plan.segments if s.audio)

    # and the video is rendered from it; after that everything is cached
    run("render")
    assert head() == v2
    after = client.get(f"/api/projects/{pid}").json()
    assert all(s["cached"] for s in after["stages"]) and after["video"]


def test_the_plan_can_be_edited_and_a_rebuild_keeps_the_edits(
    client: TestClient, services: AppServices, fakes: Providers, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    url = f"/api/projects/{pid}/plan"

    def run(step: str) -> None:
        job = client.post(f"/api/projects/{pid}/{step}").json()
        drain(services)
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "succeeded"

    err(client.get(url), 404, "not_found")
    err(client.post(f"{url}:edit", json={"base_version": 1, "ops": [_DELETE]}), 404, "not_found")
    run("script:generate")
    run("plan:build")
    v1 = client.get(url).json()
    assert [s["id"] for s in v1["segments"]] == ["seg_01", "seg_02", "seg_03"]

    ops = [
        {"op": "set_locked", "segment_id": "seg_01", "index": 0, "locked": True},
        {"op": "move_segment", "segment_id": "seg_03", "to_index": 0},
        {"op": "set_voice", "segment_id": "seg_02", "voice_id": "mine", "speed": 1.25},
    ]
    r = client.post(f"{url}:edit", json={"base_version": 1, "ops": ops})
    assert r.status_code == 200, r.text
    v2 = r.json()
    assert (v2["version"], v2["parent_version"], v2["author"], v2["id"]) == (
        2,
        1,
        "human",
        v1["id"],
    )
    assert [s["id"] for s in v2["segments"]] == ["seg_03", "seg_01", "seg_02"]
    assert v2["segments"][1]["clips"][0]["locked"] is True
    assert (v2["segments"][2]["stale"], v2["segments"][2]["voice_pinned"]) == (True, True)
    assert [v["version"] for v in client.get(f"{url}/versions").json()] == [2, 1]
    diff = client.get(f"{url}/diff", params={"a": 1, "b": 2}).json()
    assert diff["reordered"] is True
    assert {c["segment_id"]: c["status"] for c in diff["changes"]}["seg_02"] == "changed"

    # an edit made from an old version, a bad operation and an empty request are all refused
    err(client.post(f"{url}:edit", json={"base_version": 1, "ops": [_DELETE]}), 409, "conflict")
    bad = {"op": "swap_clip", "segment_id": "seg_01", "index": 9, "to": {"shot_id": "sh_0001"}}
    refused = err(
        client.post(f"{url}:edit", json={"base_version": 2, "ops": [bad]}), 422, "invalid_input"
    )
    assert "edit 1 (swap_clip)" in refused["message"]
    err(client.post(f"{url}:edit", json={"base_version": 2, "ops": []}), 422, "validation_error")
    assert client.get(url).json()["version"] == 2  # nothing half-saved

    # the next build speaks the changed voice (only that segment) and keeps everything else
    assert isinstance(fakes.tts, FakeTTS)
    spoken = len(fakes.tts.calls)
    run("plan:build")
    v3 = client.get(url).json()
    assert [c[1:] for c in fakes.tts.calls[spoken:]] == [("mine", 1.25)]
    assert (v3["version"], v3["parent_version"], v3["author"]) == (3, 2, "ai")
    assert [s["id"] for s in v3["segments"]] == ["seg_03", "seg_01", "seg_02"]
    assert v3["segments"][1] == v2["segments"][1]  # untouched, lock included
    assert v3["segments"][0] == v2["segments"][0]
    redone = v3["segments"][2]
    assert (redone["stale"], redone["voice"]["voice_id"], redone["voice_pinned"]) == (
        False,
        "mine",
        True,
    )

    # it renders in the order the person chose, and building again changes nothing
    run("render")
    run("plan:build")
    assert client.get(url).json() == v3
    after = client.get(f"/api/projects/{pid}").json()
    assert all(s["cached"] for s in after["stages"]) and after["video"]


_DELETE = {"op": "delete_segment", "segment_id": "seg_01"}


def test_one_segment_of_the_plan_can_be_previewed_quickly_and_only_once(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    url = f"/api/projects/{pid}/plan"

    def preview(segment: str, **params: Any) -> Any:
        return client.post(f"{url}/segments/{segment}:preview", params=params)

    err(preview("seg_01"), 404, "not_found")  # no plan yet
    for step in ("script:generate", "plan:build"):
        client.post(f"/api/projects/{pid}/{step}")
        drain(services)
    plan = client.get(url).json()
    err(preview("seg_99"), 404, "not_found")

    started = time.monotonic()
    first = preview("seg_01")
    elapsed = time.monotonic() - started
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["cached"] is False and body["plan_version"] == 1
    voice_ms = plan["segments"][0]["audio"]["duration_ms"]
    assert abs(body["duration_ms"] - voice_ms) <= 42
    assert elapsed < 5, f"a preview took {elapsed:.1f} s"

    served = client.get(f"/api/files/{body['file']}")
    assert served.status_code == 200 and served.headers["content-type"] == "video/mp4"
    info = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,height:format=duration",
                "-of",
                "json",
                str(services.cfg.data_dir / body["file"]),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    kinds = {s["codec_type"]: s for s in info["streams"]}
    assert kinds["video"]["height"] == 180 and "audio" in kinds  # the test film is 320x180
    assert abs(float(info["format"]["duration"]) * 1000 - body["duration_ms"]) < 100

    again = preview("seg_01").json()
    assert again["cached"] is True and again["file"] == body["file"]

    # editing seg_02 leaves seg_01's preview as it was; the edited segment is rendered anew
    second = preview("seg_02").json()
    assert second["cached"] is False and second["segment_hash"] != body["segment_hash"]
    edit = {"op": "set_locked", "segment_id": "seg_02", "index": 0, "locked": True}
    bad = {
        "op": "trim_clip",
        "segment_id": "seg_02",
        "index": 0,
        "src_in_ms": 100,
        "src_out_ms": 9000,
    }
    ok = client.post(f"{url}:edit", json={"base_version": 1, "ops": [edit, bad]})
    assert ok.status_code == 200, ok.text
    assert preview("seg_01").json()["cached"] is True
    assert preview("seg_02").json()["cached"] is False

    # a voice change leaves the segment stale: there is nothing true to show until it is built
    voice = {"op": "set_voice", "segment_id": "seg_03", "speed": 1.2}
    client.post(f"{url}:edit", json={"base_version": 2, "ops": [voice]})
    err(preview("seg_03"), 422, "invalid_input")
    assert preview("seg_03", version=1).json()["cached"] is False  # the old version still plays


def test_a_segment_offers_ranked_candidates_that_can_be_swapped_in(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    url = f"/api/projects/{pid}/plan"

    def candidates(segment: str, **params: Any) -> Any:
        return client.get(f"{url}/segments/{segment}/candidates", params=params)

    err(candidates("seg_01"), 404, "not_found")  # no plan yet
    for step in ("script:generate", "plan:build"):
        client.post(f"/api/projects/{pid}/{step}")
        drain(services)
    plan = client.get(url).json()
    err(candidates("seg_99"), 404, "not_found")
    err(candidates("seg_01", limit=0), 422, "invalid_input")

    r = candidates("seg_01")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["segment_id"], body["plan_version"]) == ("seg_01", 1)
    got = body["candidates"]
    assert got and len(got) <= 12 and body["total"] >= len(got)
    scores = [c["score"] for c in got]
    assert scores == sorted(scores, reverse=True) and all(0 <= x <= 1 for x in scores)
    shown = {c["shot_id"] for c in plan["segments"][0]["clips"]}
    assert {c["shot_id"] for c in got if c["current"]} <= shown
    assert len(candidates("seg_01", limit=1).json()["candidates"]) == 1

    # a candidate can be swapped in with the edit API
    pick = next(c for c in got if not c["current"])
    op = {"op": "swap_clip", "segment_id": "seg_01", "index": 0, "to": {"shot_id": pick["shot_id"]}}
    ok = client.post(f"{url}:edit", json={"base_version": 1, "ops": [op]})
    assert ok.status_code == 200, ok.text
    assert ok.json()["segments"][0]["clips"][0]["shot_id"] == pick["shot_id"]
    again = candidates("seg_02").json()["candidates"]
    assert all(c["used_by"] != "seg_02" for c in again)  # a segment never counts as its own user


def test_edits_and_regeneration_share_one_version_history(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    url = f"/api/projects/{pid}/script"

    def generate() -> dict[str, Any]:
        job = client.post(f"{url}:generate").json()
        drain(services)
        return client.get(f"/api/jobs/{job['id']}").json()  # type: ignore[no-any-return]

    assert generate()["status"] == "succeeded"
    v1 = client.get(url).json()
    assert generate()["status"] == "succeeded"  # same draft again: nothing new is stored
    assert client.get(url).json()["version"] == 1

    # a person edits; regenerating then stacks the AI draft on top of the edit, nothing is lost
    edit = {k: v1[k] for k in ("params", "outline", "segments", "annotations")}
    edit["segments"][0]["text"] = "我改过的开场白。"
    saved = client.put(url, json={**edit, "base_version": 1}).json()
    assert (saved["version"], saved["author"]) == (2, "human")
    assert generate()["status"] == "succeeded"
    head = client.get(url).json()
    assert (head["version"], head["author"], head["parent_version"]) == (3, "ai", 2)
    assert head["segments"][0]["text"] != "我改过的开场白。"
    assert (
        client.get(url, params={"version": 2}).json()["segments"][0]["text"] == "我改过的开场白。"
    )

    # a person saves while a generation is queued: the job must not bury that edit
    job = client.post(f"{url}:generate").json()
    edit["segments"][0]["text"] = "又一次修改。"
    assert client.put(url, json={**edit, "base_version": 3}).status_code == 200
    drain(services)
    failed = client.get(f"/api/jobs/{job['id']}").json()
    assert (
        failed["status"] == "failed" and "changed while it was being generated" in failed["error"]
    )
    assert client.get(url).json()["version"] == 4 and client.get(url).json()["author"] == "human"


def test_one_segment_can_be_rewritten_into_a_new_version(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    url = f"/api/projects/{pid}/script"
    err(
        client.post(f"{url}/segments/seg_01:rewrite", json={"instruction": "x", "base_version": 1}),
        404,
        "not_found",
    )  # no script yet
    client.post(f"{url}:generate")
    drain(services)
    v1 = client.get(url).json()
    assert [a["segment_id"] for a in v1["annotations"]] == ["seg_01"]  # the critic flagged it

    rewrite = f"{url}/segments/seg_01:rewrite"
    job = client.post(rewrite, json={"instruction": "更口语化", "base_version": 1})
    assert job.status_code == 202 and job.json()["stage"] == "creation.rewrite"
    drain(services)
    assert client.get(f"/api/jobs/{job.json()['id']}").json()["status"] == "succeeded"

    v2 = client.get(url).json()
    assert (v2["version"], v2["author"], v2["parent_version"]) == (2, "ai", 1)
    assert v2["segments"][0]["text"] != v1["segments"][0]["text"]
    assert v2["segments"][0]["text"].startswith("改")
    assert v2["segments"][1:] == v1["segments"][1:]  # nothing else moved
    assert v2["annotations"] == []  # the verdict on the old words is gone
    diff = client.get(f"{url}/diff", params={"a": 1, "b": 2}).json()
    assert [(c["segment_id"], c["status"]) for c in diff["changes"]] == [
        ("seg_01", "changed"),
        ("seg_02", "unchanged"),
        ("seg_03", "unchanged"),
    ]

    err(client.post(rewrite, json={"instruction": "x", "base_version": 1}), 409, "conflict")
    err(
        client.post(f"{url}/segments/seg_99:rewrite", json={"instruction": "x", "base_version": 2}),
        404,
        "not_found",
    )
    err(
        client.post(rewrite, json={"instruction": "  ", "base_version": 2}),
        422,
        "invalid_input",
    )
    err(client.post(rewrite, json={"instruction": "", "base_version": 2}), 422, "validation_error")
    err(client.post(rewrite, json={"instruction": "x"}), 422, "validation_error")


def test_a_rewrite_does_not_bury_an_edit_made_while_it_waited(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    url = f"/api/projects/{pid}/script"
    client.post(f"{url}:generate")
    drain(services)
    v1 = client.get(url).json()

    job = client.post(
        f"{url}/segments/seg_02:rewrite", json={"instruction": "短一点", "base_version": 1}
    )
    edit = {k: v1[k] for k in ("params", "outline", "segments", "annotations")}
    edit["segments"][2]["text"] = "我在排队期间改了第三段。"
    assert client.put(url, json={**edit, "base_version": 1}).status_code == 200
    drain(services)

    failed = client.get(f"/api/jobs/{job.json()['id']}").json()
    assert (
        failed["status"] == "failed"
        and "changed while seg_02 was being rewritten" in failed["error"]
    )
    head = client.get(url).json()
    assert (head["version"], head["author"]) == (2, "human")
    assert head["segments"][2]["text"] == "我在排队期间改了第三段。"


def test_without_a_configured_critic_the_script_is_stored_unreviewed(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    tasks = {k: v for k, v in cfg.tasks.items() if k != "script_critic"}
    rooted = cfg.model_copy(update={"media_roots": [movie.parent], "tasks": tasks})
    with (
        AppServices(rooted, providers=fakes) as services,
        TestClient(create_app(services), raise_server_exceptions=False) as client,
    ):
        asset = import_movie(client, movie)
        pid = client.post(
            "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
        ).json()["id"]
        job = client.post(f"/api/projects/{pid}/script:generate").json()
        drain(services)
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "succeeded"
        assert client.get(f"/api/projects/{pid}/script").json()["annotations"] == []


def test_the_script_follows_the_outline_a_person_edited(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    pid = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()["id"]
    err(client.get(f"/api/projects/{pid}/outline"), 404, "not_found")
    assert client.post(f"/api/projects/{pid}/outline:generate").status_code == 202
    drain(services)

    outline = client.get(f"/api/projects/{pid}/outline").json()
    assert outline["edited"] is False and [b["beat"] for b in outline["outline"]["beats"]] == [
        "hook",
        "development",
        "ending",
    ]
    first = outline["outline"]["beats"][0]
    edit = [{**first, "beat": "opening", "target_s": 5, "focus": "我改的"}]
    edit += [{**b, "target_s": 5} for b in outline["outline"]["beats"][1:]]
    assert client.put(f"/api/projects/{pid}/outline", json={"beats": edit}).status_code == 200

    client.post(f"/api/projects/{pid}/script:generate")
    drain(services)
    script = client.get(f"/api/projects/{pid}/script").json()
    assert [b["beat"] for b in script["outline"]] == ["opening", "development", "ending"]
    assert script["segments"][0]["beat"] == "opening"

    # dropping the edit goes back to the generated outline, and the next script follows it
    client.delete(f"/api/projects/{pid}/outline")
    client.post(f"/api/projects/{pid}/script:generate")
    drain(services)
    assert client.get(f"/api/projects/{pid}/script").json()["segments"][0]["beat"] == "hook"


# --- jobs -------------------------------------------------------------------------------


def test_job_control_over_http(client: TestClient, services: AppServices, movie: Path) -> None:
    asset = import_movie(client, movie)
    job = client.post(f"/api/assets/{asset['id']}/analyze").json()

    assert [j["id"] for j in client.get("/api/jobs", params={"status": "queued"}).json()] == [
        job["id"]
    ]
    assert client.get("/api/jobs", params={"lane": "gpu"}).json() == []
    err(client.get("/api/jobs", params={"status": "bogus"}), 422, "validation_error")

    err(client.post(f"/api/jobs/{job['id']}:retry"), 409, "conflict")  # not failed
    canceled = client.post(f"/api/jobs/{job['id']}:cancel")
    assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
    assert client.post(f"/api/jobs/{job['id']}:cancel").json()["status"] == "canceled"  # idempotent
    drain(services)  # nothing left to run
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "canceled"
    assert client.get(f"/api/jobs/{job['id']}/log").text == ""

    err(client.get("/api/jobs/job_missing"), 404, "not_found")
    err(client.get("/api/jobs/job_missing/log"), 404, "not_found")
    err(client.post("/api/jobs/job_missing:cancel"), 404, "not_found")
    err(client.post("/api/jobs/job_missing:retry"), 404, "not_found")


# --- error model ------------------------------------------------------------------------


def test_unknown_route_and_wrong_method_use_the_error_shape(client: TestClient) -> None:
    err(client.get("/api/nothing-here"), 404, "not_found")
    err(client.delete("/api/assets"), 405, "method_not_allowed")


def test_unexpected_failures_are_500_without_leaking_details(
    cfg: AppConfig, fakes: Providers, services: AppServices
) -> None:
    def boom(**_: Any) -> Any:
        raise RuntimeError("secret internal detail /home/x/key")

    services.jobs.list = boom  # type: ignore[method-assign,assignment]
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        e = err(c.get("/api/jobs"), 500, "internal")
    assert e["message"] == "internal error" and "secret" not in str(e)
