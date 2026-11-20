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
    delivery = done.client.get(f"/deliveries/{body['delivery_ids'][0]}").json()
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


def test_worker_retries_then_fails(env: Env) -> None:
    r = env.client.post("/sessions", files={"file": ("bad.mp4", b"not a video", "video/mp4")})
    sid = r.json()["session"]["id"]
    for _ in range(env.settings.jobs.max_attempts):
        assert env.worker.run_once() is True
    assert env.worker.run_once() is False
    body = env.client.get(f"/sessions/{sid}").json()
    assert body["session"]["status"] == "failed" and body["result"] is None
    assert body["job"]["attempts"] == env.settings.jobs.max_attempts
    job = env.repo.latest_job_for_session(sid)
    assert job is not None and job.last_error is not None and "Traceback" in job.last_error


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
