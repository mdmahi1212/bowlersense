"""Phase 12 gate: the FastAPI app end to end with the SQLite DB and the worker (TestClient)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from bowlsense.api import create_app
from bowlsense.config import Settings, load_settings
from bowlsense.db.repo import Repository
from bowlsense.db.session import get_engine
from bowlsense.local_settings import LocalSettings
from bowlsense.storage import LocalStorage
from bowlsense.worker import Worker
from synth.skeleton_gen import SynthParams, generate, render_to_video

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, bytes]:
    """One synthetic delivery (video bytes, PoseSequence JSON bytes)."""
    d = tmp_path_factory.mktemp("clip")
    r = generate(SynthParams(fps=120, height_px=300, elbow_extension_deg=22.0))
    video = render_to_video(r.sequence, d / "one.mp4")
    return video.read_bytes(), r.sequence.model_dump_json().encode()


def _isolated(tmp: Path) -> Settings:
    s = load_settings()
    storage = s.storage.model_copy(update={"root": tmp / "storage", "db_path": tmp / "db.sqlite"})
    return s.model_copy(update={"storage": storage})


class Env:
    """App + worker sharing one temporary database and storage."""

    def __init__(self, tmp: Path) -> None:
        self.settings = _isolated(tmp)
        self.client = TestClient(create_app(self.settings))
        engine = get_engine(str(self.settings.storage.db_file))
        self.repo = Repository(engine)
        self.storage = LocalStorage(self.settings.storage.root_path)
        self.worker = Worker(self.settings, self.repo, self.storage)
        self.session_id = ""
        self.bowler_id = ""

    def upload(self, clip: tuple[bytes, bytes], **form: str) -> dict[str, object]:
        video, pose = clip
        r = self.client.post(
            "/sessions",
            files={
                "file": ("my clip.mp4", video, "video/mp4"),
                "pose_file": ("pose.json", pose, "application/json"),
            },
            data=form,
        )
        assert r.status_code == 201, r.text
        body: dict[str, object] = r.json()
        return body


@pytest.fixture()
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path)
    local = e.client.put(
        "/settings", json={"coach_model": "llama3.1:8b", "pose_backend": "rtm", "use_llm": False}
    )
    assert local.status_code == 200
    yield e
    e.client.close()


@pytest.fixture(scope="module")
def done(tmp_path_factory: pytest.TempPathFactory, clip: tuple[bytes, bytes]) -> Iterator[Env]:
    """A finished session (one upload processed by the worker)."""
    e = Env(tmp_path_factory.mktemp("done"))
    e.client.put("/settings", json={"coach_model": "llama3.1:8b", "pose_backend": "rtm",
                                    "use_llm": False})  # fmt: skip
    bowler = e.client.post("/bowlers", json={"name": "Test Bowler", "bowling_arm": "right"}).json()
    session = e.upload(clip, bowler_id=bowler["id"])["session"]
    assert isinstance(session, dict)
    e.session_id = session["id"]
    e.bowler_id = bowler["id"]
    assert e.worker.run_once() is True
    yield e
    e.client.close()


def _sid(e: Env) -> str:
    return e.session_id


# --- health / settings --------------------------------------------------------------------------


def test_health(env: Env) -> None:
    body = env.client.get("/health").json()
    assert body["status"] == "ok" and set(body["models"]) >= {"yolo", "pose_rtm"}
    assert {"torch_cuda", "onnxruntime_cuda", "enabled"} <= set(body["gpu"])


def test_settings_roundtrip(env: Env) -> None:
    body = env.client.get("/settings").json()
    assert body["settings"]["use_llm"] is False and body["storage_bytes"] >= 0
    bad = env.client.put("/settings", json={"coach_model": "x", "pose_backend": "nope"})
    assert bad.status_code == 422


@respx.mock
def test_llm_models_listed_and_unavailable(env: Env) -> None:
    url = env.settings.coach.ollama_url.rstrip("/") + "/api/tags"
    route = respx.get(url).mock(
        return_value=httpx.Response(200, json={"models": [{"name": "a:1b"}]})
    )
    assert env.client.get("/llm/models").json()["models"] == ["a:1b"]
    route.mock(side_effect=httpx.ConnectError("refused"))
    body = env.client.get("/llm/models").json()
    assert body["available"] is False and body["models"] == []


def test_cors_only_local_web_app(env: Env) -> None:
    port = env.settings.server.web_port
    ok = env.client.get("/health", headers={"Origin": f"http://localhost:{port}"})
    assert ok.headers.get("access-control-allow-origin") == f"http://localhost:{port}"
    bad = env.client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


# --- uploads ------------------------------------------------------------------------------------


def test_upload_rejects_bad_types(env: Env) -> None:
    r = env.client.post("/sessions", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    r = env.client.post("/sessions", files={"file": ("x.png", b"\x89PNG", "image/png")})
    assert r.status_code == 415 and "/analyze/image" in r.json()["detail"]
    assert env.repo.count_rows("sessions") == 0
    assert not any((env.storage.root / "raw").iterdir())


def test_upload_too_large(env: Env, clip: tuple[bytes, bytes]) -> None:
    api = env.settings.api.model_copy(update={"max_upload_mb": 1})
    small = TestClient(create_app(env.settings.model_copy(update={"api": api})))
    r = small.post("/sessions", files={"file": ("big.mp4", b"0" * (1024 * 1024 + 1), "video/mp4")})
    assert r.status_code == 413
    assert not any((env.storage.root / "raw").iterdir())


def test_upload_queues_job_and_stores_by_uuid(env: Env, clip: tuple[bytes, bytes]) -> None:
    body = env.upload(clip)
    session = body["session"]
    assert isinstance(session, dict)
    assert session["status"] == "queued" and session["original_name"] == "my clip.mp4"
    assert "my clip" not in session["video_path"] and session["video_path"].startswith("raw/")
    job = body["job"]
    assert isinstance(job, dict) and job["status"] == "queued"
    listed = env.client.get("/sessions").json()
    assert [s["id"] for s in listed] == [session["id"]]


def test_unknown_bowler_rejected(env: Env, clip: tuple[bytes, bytes]) -> None:
    r = env.client.post(
        "/sessions", files={"file": ("a.mp4", clip[0], "video/mp4")}, data={"bowler_id": "nope"}
    )
    assert r.status_code == 404 and env.repo.count_rows("sessions") == 0


# --- end to end ---------------------------------------------------------------------------------


def test_gate_upload_to_results(done: Env) -> None:
    body = done.client.get(f"/sessions/{_sid(done)}").json()
    assert body["session"]["status"] == "done" and body["job"]["progress"] == 100
    result = body["result"]
    assert len(result["deliveries"]) == 1 and result["deliveries"][0]["legality"]["flagged"]
    assert result["artifacts"]["overlay"].startswith(f"outputs/{_sid(done)}/")
    assert result["workload"]["balls_this_session"] == 1
    assert len(body["delivery_ids"]) == 1
    out = done.client.get(f"/deliveries/{body['delivery_ids'][0]}").json()
    assert out["session_id"] == _sid(done) and out["fps"] == 120 and out["delivery_count"] == 1
    assert out["overlay"] == result["artifacts"]["overlay"]
    delivery = out["delivery"]
    assert delivery["delivery_idx"] == 0 and delivery["angles"][0]["name"] == "elbow"
    assert all(k.startswith("outputs/") for k in delivery["keyframes"].values())
    assert done.repo.count_rows("faults") >= 0 and done.repo.count_rows("delivery_metrics") == 3


def test_sse_stream_reports_done(done: Env) -> None:
    with done.client.stream("GET", f"/sessions/{_sid(done)}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1]["status"] == "done" and events[-1]["progress"] == 100


def test_files_with_range_and_traversal(done: Env) -> None:
    overlay = done.client.get(f"/sessions/{_sid(done)}").json()["result"]["artifacts"]["overlay"]
    full = done.client.get(f"/files/{overlay}")
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    part = done.client.get(f"/files/{overlay}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert part.headers["content-range"].startswith("bytes 0-99/")
    assert done.client.get("/files/raw/anything.mp4").status_code == 400  # outputs/ only
    assert done.client.get("/files/outputs/../../etc/passwd").status_code in (400, 404)
    assert done.client.get("/files/outputs/missing.png").status_code == 404


def test_pdf_and_coach_chat(done: Env) -> None:
    pdf = done.client.get(f"/sessions/{_sid(done)}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    r = done.client.post(f"/sessions/{_sid(done)}/coach", json={"message": "How is my elbow?"})
    assert r.status_code == 200 and r.json()["source"] == "template"
    history = done.client.get(f"/sessions/{_sid(done)}/coach").json()
    assert [m["role"] for m in history[-2:]] == ["user", "assistant"]


def test_bowler_workload(done: Env) -> None:
    body = done.client.get(f"/bowlers/{done.bowler_id}/workload").json()
    assert len(body["points"]) == 1 and body["points"][0]["balls"] == 1
    assert body["points"][0]["acwr"] is None and body["safe_band"] == [0.8, 1.3]
    filtered = done.client.get("/sessions", params={"bowler_id": done.bowler_id}).json()
    assert [s["id"] for s in filtered] == [_sid(done)]


def test_calibration_requeues_and_reuses_pose(done: Env) -> None:
    bad = done.client.post(f"/sessions/{_sid(done)}/calibration", json={"points": [[0, 0]] * 4})
    assert bad.status_code == 422
    pts = [[100.0, 400.0], [300.0, 400.0], [260.0, 200.0], [140.0, 200.0]]
    r = done.client.post(f"/sessions/{_sid(done)}/calibration", json={"points": pts})
    assert r.status_code == 200 and r.json()["session"]["status"] == "queued"
    assert r.json()["session"]["options"]["reuse_pose"] is True
    assert done.worker.run_once() is True
    after = done.client.get(f"/sessions/{_sid(done)}").json()
    assert after["session"]["status"] == "done" and after["session"]["calibration"] == pts


# --- failures and deletion ----------------------------------------------------------------------


def test_bad_input_fails_at_once(env: Env) -> None:
    """An unreadable video fails the same way every time, so it isn't retried."""
    r = env.client.post("/sessions", files={"file": ("bad.mp4", b"not a video", "video/mp4")})
    sid = r.json()["session"]["id"]
    assert env.worker.run_once() is True
    assert env.worker.run_once() is False
    body = env.client.get(f"/sessions/{sid}").json()
    assert body["session"]["status"] == "failed" and body["result"] is None
    assert body["job"]["attempts"] == 1
    job = env.repo.latest_job_for_session(sid)
    assert job is not None and job.last_error is not None and "Traceback" in job.last_error


def test_unexpected_errors_are_retried(
    env: Env, clip: tuple[bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    import bowlsense.worker as worker_mod

    def crash(*args: object, **kwargs: object) -> object:
        raise RuntimeError("GPU out of memory")

    monkeypatch.setattr(worker_mod, "run_pipeline", crash)
    session = env.upload(clip)["session"]
    assert isinstance(session, dict)
    for _ in range(env.settings.jobs.max_attempts):
        assert env.worker.run_once() is True
    assert env.worker.run_once() is False
    body = env.client.get(f"/sessions/{session['id']}").json()
    assert body["session"]["status"] == "failed"
    assert body["job"]["attempts"] == env.settings.jobs.max_attempts


def test_delete_session_removes_rows_and_files(env: Env, clip: tuple[bytes, bytes]) -> None:
    session = env.upload(clip)["session"]
    assert isinstance(session, dict)
    sid = session["id"]
    assert env.worker.run_once()
    out = env.storage.root / "outputs" / str(sid)
    assert out.is_dir()
    assert env.client.delete(f"/sessions/{sid}").status_code == 200
    assert not out.exists() and not any((env.storage.root / "raw").iterdir())
    assert env.repo.count_rows("deliveries") == 0 and env.repo.count_rows("jobs") == 0
    assert env.client.get(f"/sessions/{sid}").status_code == 404


def test_delete_all_data_needs_confirmation(env: Env, clip: tuple[bytes, bytes]) -> None:
    env.client.post("/bowlers", json={"name": "B"})
    env.upload(clip)
    no = env.client.request("DELETE", "/data", json={"confirm": "yes"})
    assert no.status_code == 400 and env.repo.count_rows("sessions") == 1
    ok = env.client.request("DELETE", "/data", json={"confirm": "DELETE"})
    assert ok.status_code == 200
    assert env.repo.count_rows("sessions") == 0 and env.repo.count_rows("bowlers") == 0
    assert not any((env.storage.root / "raw").iterdir())


def test_bowler_validation(env: Env) -> None:
    assert env.client.post("/bowlers", json={"name": ""}).status_code == 422
    assert env.client.post("/bowlers", json={"name": "X", "height_cm": 10}).status_code == 422
    ok = env.client.post(
        "/bowlers", json={"name": "Ann", "bowling_arm": "left", "dob": "2005-02-03"}
    )
    assert ok.status_code == 201 and ok.json()["dob"] == "2005-02-03"
    assert [b["name"] for b in env.client.get("/bowlers").json()] == ["Ann"]
    assert env.client.get("/bowlers/nope/workload").status_code == 404


def test_raw_video_drills_and_references(done: Env) -> None:
    video = done.client.get(f"/sessions/{_sid(done)}/video", headers={"Range": "bytes=0-9"})
    assert video.status_code == 206 and len(video.content) == 10
    assert done.client.get("/sessions/nope/video").status_code == 404
    drills = done.client.get("/drills").json()
    assert len(drills) == 10 and all(2 <= len(v) <= 3 for v in drills.values())
    refs = done.client.get("/references").json()
    assert len(refs["references"]) >= 2 and ["l_shoulder", "r_shoulder"] in refs["edges"]
    assert all(r["name"].startswith("Reference: ") for r in refs["references"])


class _PhotoBackend:
    name = "fake"

    def estimate(self, frame: object, bbox: object) -> dict[str, object]:
        r = generate(SynthParams(height_px=400.0))
        return dict(r.sequence.frames[r.phases.release].keypoints)


def test_analyze_image(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    import bowlsense.image_mode as im

    monkeypatch.setattr(im, "detect_person", lambda img, s: (10.0, 10.0, 600.0, 700.0))
    monkeypatch.setattr(im, "load_backend", lambda *a, **k: _PhotoBackend())
    ok, png = cv2.imencode(".png", np.full((720, 1280, 3), 30, np.uint8))
    assert ok
    r = env.client.post(
        "/analyze/image",
        files={"file": ("photo.png", png.tobytes(), "image/png")},
        data={"phase": "release"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["phase"] == "release" and body["phase_source"] == "user"
    assert {c["code"] for c in body["checks"]} >= {"FRONT_KNEE", "ARM_SLOT"}
    assert body["annotated"].startswith("outputs/images/")
    assert env.client.get(f"/files/{body['annotated']}").status_code == 200
    assert not any((env.storage.root / "raw").iterdir())  # the original photo isn't kept
    bad = env.client.post("/analyze/image", files={"file": ("x.mp4", b"0", "video/mp4")})
    assert bad.status_code == 415
    monkeypatch.setattr(im, "detect_person", _no_person)
    nobody = env.client.post(
        "/analyze/image", files={"file": ("p.png", png.tobytes(), "image/png")}
    )
    assert nobody.status_code == 422 and "No person" in nobody.json()["detail"]
    assert len(list((env.storage.root / "outputs" / "images").iterdir())) == 1  # the first one


def _no_person(img: object, s: object) -> object:
    from bowlsense.errors import NoBowlerDetectedError

    raise NoBowlerDetectedError("No person found in the image. Use a photo of the bowler.")


def test_worker_survives_session_deleted_mid_job(
    env: Env, clip: tuple[bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a session while it is analysed must not crash the worker (it skips the job)."""
    import threading

    import bowlsense.worker as worker_mod

    session = env.upload(clip)["session"]
    assert isinstance(session, dict)
    sid = session["id"]

    def deleting_pipeline(*args: object, **kwargs: object) -> object:
        env.client.delete(f"/sessions/{sid}")  # the user deletes it mid-run
        progress = kwargs["progress"]
        assert callable(progress)
        progress(50.0, "pose")  # the next progress write finds no job
        raise AssertionError("unreachable")

    monkeypatch.setattr(worker_mod, "run_pipeline", deleting_pipeline)
    assert env.worker.run_once() is True  # no exception escapes
    assert env.client.get(f"/sessions/{sid}").status_code == 404
    stop = threading.Event()
    calls = []

    def boom() -> bool:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("db hiccup")
        stop.set()
        return False

    monkeypatch.setattr(env.worker, "run_once", boom)
    env.worker.run_forever(stop)  # keeps polling after an unexpected error
    assert len(calls) == 2


def test_pick_the_bowler(env: Env, clip: tuple[bytes, bytes]) -> None:
    from bowlsense.schemas import PeopleTracks, TrackedPerson

    session = env.upload(clip)["session"]
    assert isinstance(session, dict)
    sid = session["id"]
    busy = env.client.post(f"/sessions/{sid}/bowler", json={"track_id": 2})
    assert busy.status_code == 409  # still queued
    assert env.worker.run_once()
    assert env.client.get(f"/sessions/{sid}/people").status_code == 404  # keypoints imported
    people = PeopleTracks(
        width=640, height=360, fps=120.0, frame_count=3, chosen_track_id=1, chosen_by="auto",
        people=[TrackedPerson(track_id=t, frames=[0, 1, 2], boxes=[(0, 0, 10, 20)] * 3)
                for t in (1, 2)],
    )  # fmt: skip
    (env.storage.root / "outputs" / sid / "people.json").write_text(people.model_dump_json())
    got = env.client.get(f"/sessions/{sid}/people").json()
    assert got["chosen_track_id"] == 1 and [p["track_id"] for p in got["people"]] == [1, 2]
    assert env.client.post(f"/sessions/{sid}/bowler", json={"track_id": 7}).status_code == 422
    r = env.client.post(f"/sessions/{sid}/bowler", json={"track_id": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session"]["status"] == "queued"
    assert body["session"]["options"]["bowler_track_id"] == 2
    assert "reuse_pose" not in body["session"]["options"]
    job = env.repo.claim_next_job()
    assert job is not None
    assert env.worker.options_for(job, load_local_settings_for(env)).bowler_track_id == 2


def load_local_settings_for(env: Env) -> LocalSettings:
    from bowlsense.local_settings import load_local_settings

    return load_local_settings(env.settings)


def test_user_sets_release_frame(done: Env) -> None:
    did = done.client.get(f"/sessions/{_sid(done)}").json()["delivery_ids"][0]
    before = done.client.get(f"/deliveries/{did}").json()["delivery"]
    frame = before["phases"]["release"] + 3
    r = done.client.post(f"/deliveries/{did}/release", json={"frame": frame})
    assert r.status_code == 200, r.text
    after = r.json()["delivery"]
    assert after["phases"]["release"] == frame and after["release_source"] == "user"
    assert after["biomechanics"] and after["legality"]["verdict"] in ("legal", "not_legal")
    again = done.client.get(f"/deliveries/{did}").json()["delivery"]  # stored, same id
    assert again["phases"]["release"] == frame
    bad = done.client.post(f"/deliveries/{did}/release", json={"frame": 99999})
    assert bad.status_code == 422


def test_release_clip_mp4_and_gif(done: Env) -> None:
    did = done.client.get(f"/sessions/{_sid(done)}").json()["delivery_ids"][0]
    mp4 = done.client.get(f"/deliveries/{did}/clip")
    assert mp4.status_code == 200, mp4.text
    assert mp4.headers["content-type"] == "video/mp4" and mp4.content[4:8] == b"ftyp"
    assert "attachment" in mp4.headers["content-disposition"]
    gif = done.client.get(f"/deliveries/{did}/clip", params={"format": "gif"})
    assert gif.status_code == 200 and gif.content[:6] in (b"GIF89a", b"GIF87a")
    assert done.client.get(f"/deliveries/{did}/clip", params={"format": "avi"}).status_code == 422


def test_annotations_roundtrip(done: Env) -> None:
    did = done.client.get(f"/sessions/{_sid(done)}").json()["delivery_ids"][0]
    assert done.client.get(f"/deliveries/{did}/annotations").json() == {"annotations": []}
    body = {
        "annotations": [
            {"kind": "angle", "frame": 12, "points": [[1, 2], [3, 4], [5, 6]], "label": "trunk"},
            {"kind": "line", "frame": 3, "points": [[0, 0], [10, 10]], "label": ""},
        ]
    }
    assert done.client.put(f"/deliveries/{did}/annotations", json=body).status_code == 200
    assert done.client.get(f"/deliveries/{did}/annotations").json() == body
    bad = {"annotations": [{"kind": "angle", "frame": 1, "points": [[0, 0]]}]}
    assert done.client.put(f"/deliveries/{did}/annotations", json=bad).status_code == 422


def test_bowler_progress(done: Env) -> None:
    body = done.client.get(f"/bowlers/{done.bowler_id}/progress").json()
    assert len(body["points"]) == 1
    p = body["points"][0]
    assert p["session_id"] == _sid(done) and p["deliveries"] == len(p["items"]) == 1
    assert (
        0 <= p["technique"] <= 100 and body["best"]["delivery_id"] == p["items"][0]["delivery_id"]
    )
    assert done.client.get("/bowlers/nope/progress").status_code == 404


def test_notes_need_text_and_one_point(done: Env) -> None:
    did = done.client.get(f"/sessions/{_sid(done)}").json()["delivery_ids"][0]
    ok = {"annotations": [{"kind": "note", "frame": 4, "points": [[5, 5]], "label": "Head falls"}]}
    assert done.client.put(f"/deliveries/{did}/annotations", json=ok).status_code == 200
    for bad in (
        {"kind": "note", "frame": 4, "points": [[5, 5]], "label": "  "},
        {"kind": "line", "frame": 4, "points": [[5, 5], [1, 1], [2, 2]]},
    ):
        r = done.client.put(f"/deliveries/{did}/annotations", json={"annotations": [bad]})
        assert r.status_code == 422


def test_drill_plan_and_goals(done: Env) -> None:
    bid = done.bowler_id
    assert done.client.get(f"/bowlers/{bid}/plan").json() is None
    plan = done.client.post(f"/bowlers/{bid}/plan").json()
    assert len(plan["sessions"]) == 4 * 3 and plan["faults"]
    assert all(it["drill"] and not it["done"] for s in plan["sessions"] for it in s["items"])
    plan["sessions"][0]["items"][0]["done"] = True
    assert done.client.put(f"/bowlers/{bid}/plan", json=plan).status_code == 200
    assert done.client.get(f"/bowlers/{bid}/plan").json()["sessions"][0]["items"][0]["done"]
    goals = {"goals": [{"metric": "technique", "op": "above", "target": 75, "label": "Tidy up"}]}
    assert done.client.put(f"/bowlers/{bid}/goals", json=goals).status_code == 200
    assert done.client.get(f"/bowlers/{bid}/goals").json() == goals
    bad = {"goals": [{"metric": "speed", "op": "above", "target": 1}]}
    assert done.client.put(f"/bowlers/{bid}/goals", json=bad).status_code == 422
    assert done.client.post("/bowlers/nope/plan").status_code == 404


def test_spell_limit_for_juniors(env: Env) -> None:
    young = env.client.post(
        "/bowlers", json={"name": "Junior", "bowling_arm": "right", "dob": "2013-06-01"}
    ).json()
    adult = env.client.post(
        "/bowlers", json={"name": "Senior", "bowling_arm": "right", "dob": "1990-01-01"}
    ).json()
    limit = env.client.get(f"/bowlers/{young['id']}/workload").json()["spell_limit"]
    assert limit is not None and limit["overs_per_spell"] in (5, 6, 7) and limit["age"] >= 12
    assert env.client.get(f"/bowlers/{adult['id']}/workload").json()["spell_limit"] is None


def test_readiness_stored_with_the_session(env: Env, clip: tuple[bytes, bytes]) -> None:
    s = env.upload(clip, sleep="4", soreness="2", effort="7")["session"]
    assert isinstance(s, dict)
    assert s["options"]["readiness"] == {"sleep": 4, "soreness": 2, "effort": 7}
    r = env.client.post(
        "/sessions",
        files={"file": ("c.mp4", clip[0], "video/mp4")},
        data={"soreness": "9"},
    )
    assert r.status_code == 422


def test_in_app_recording_is_made_constant_rate(env: Env, clip: tuple[bytes, bytes]) -> None:
    from bowlsense.io.video import probe_video

    s = env.upload(clip, capture_fps="30")["session"]
    assert isinstance(s, dict)
    video = env.settings.storage.root_path / str(s["video_path"])
    assert video.suffix == ".mp4" and video.is_file()
    assert abs(probe_video(video).fps - 30.0) < 0.5
    assert not list(video.parent.glob("*.rec.mp4"))  # temp file cleaned up


def test_highlight_reel_and_poster(done: Env) -> None:
    reel = done.client.get(f"/sessions/{_sid(done)}/reel.mp4")
    assert reel.status_code == 200, reel.text
    assert reel.content[4:8] == b"ftyp" and "attachment" in reel.headers["content-disposition"]
    did = done.client.get(f"/sessions/{_sid(done)}").json()["delivery_ids"][0]
    png = done.client.get(f"/deliveries/{did}/poster.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_pdf_gets_the_live_appendix(done: Env) -> None:
    from io import BytesIO

    from pypdf import PdfReader

    sid = _sid(done)
    did = done.client.get(f"/sessions/{sid}").json()["delivery_ids"][0]
    clear = done.client.put(f"/deliveries/{did}/annotations", json={"annotations": []})
    assert clear.status_code == 200  # other tests may have saved marks
    before = len(PdfReader(BytesIO(done.client.get(f"/sessions/{sid}/report.pdf").content)).pages)
    marks = {"annotations": [{"kind": "angle", "frame": 2, "points": [[0, 0], [5, 5], [10, 0]]}]}
    assert done.client.put(f"/deliveries/{did}/annotations", json=marks).status_code == 200
    pdf = done.client.get(f"/sessions/{sid}/report.pdf")
    assert pdf.status_code == 200
    reader = PdfReader(BytesIO(pdf.content))
    assert len(reader.pages) == before + 1
    assert "Angle 90.0 deg" in reader.pages[-1].extract_text()


def test_delivery_has_no_ball_and_speed_fields(done: Env) -> None:
    did = done.client.get(f"/sessions/{_sid(done)}").json()["delivery_ids"][0]
    d = done.client.get(f"/deliveries/{did}").json()["delivery"]
    assert "speed_estimate_kmh" in d
    assert d["no_ball"] is None or d["no_ball"]["status"] in ("legal", "no_ball", "not_found")
