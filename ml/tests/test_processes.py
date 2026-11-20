"""Phase 13 gate: the real API server and the real worker as two processes on one SQLite file.

Uploads, bowler/chat writes and SSE reads hit the API while the worker claims jobs, writes
progress and stores results, with no ``database is locked`` errors.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import yaml

from bowlsense.config import DEFAULT_CONFIG_PATH
from synth.skeleton_gen import SynthParams, generate, render_to_video

ML_DIR = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path, list[Path]]]:
    """API + worker processes on a temporary config, DB and storage."""
    tmp = tmp_path_factory.mktemp("stack")
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["storage"].update(root=str(tmp / "storage"), db_path=str(tmp / "db.sqlite"))
    cfg["storage"]["models_dir"] = str((DEFAULT_CONFIG_PATH.parents[1] / "models").resolve())
    cfg["jobs"]["poll_interval_s"] = 0.2
    cfg_file = tmp / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp / "settings.json").write_text(
        json.dumps({"coach_model": "llama3.1:8b", "pose_backend": "rtm", "use_llm": False}),
        encoding="utf-8",
    )
    port = _free_port()
    env = {**os.environ, "BOWLSENSE_CONFIG": str(cfg_file), "BOWLSENSE_API_PORT": str(port),
           "BOWLSENSE_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}  # fmt: skip
    logs = [tmp / "api.log", tmp / "worker.log"]
    procs = []
    for module, log in zip(("bowlsense.api", "bowlsense.worker"), logs, strict=True):
        with log.open("w", encoding="utf-8") as fh:
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", module],
                    cwd=ML_DIR,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
            )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.3)
    else:
        for p in procs:
            p.kill()
        pytest.fail("API did not start:\n" + logs[0].read_text(encoding="utf-8"))
    try:
        yield base, tmp, logs
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


def test_gate_api_and_worker_processes_share_the_db(
    stack: tuple[str, Path, list[Path]], no_offline_guard: None
) -> None:
    base, tmp, logs = stack
    r = generate(SynthParams(fps=120, height_px=300))
    video = render_to_video(r.sequence, tmp / "clip.mp4").read_bytes()
    pose = r.sequence.model_dump_json().encode()
    with httpx.Client(base_url=base, timeout=30) as client:
        bowler = client.post("/bowlers", json={"name": "Proc"}).json()
        ids = []
        for i in range(3):
            res = client.post(
                "/sessions",
                files={
                    "file": (f"c{i}.mp4", video, "video/mp4"),
                    "pose_file": ("p.json", pose, "application/json"),
                },
                data={"bowler_id": bowler["id"]},
            )
            assert res.status_code == 201, res.text
            ids.append(res.json()["session"]["id"])
        # keep writing through the API while the worker is busy
        deadline, writes = time.monotonic() + 300, 0
        statuses: list[str] = []
        while time.monotonic() < deadline:
            client.post("/bowlers", json={"name": f"extra {writes}"}).raise_for_status()
            writes += 1
            statuses = [client.get(f"/sessions/{i}").json()["session"]["status"] for i in ids]
            if all(s in ("done", "failed") for s in statuses):
                break
            time.sleep(0.2)
        assert statuses == ["done"] * 3, statuses
        with client.stream("GET", f"/sessions/{ids[0]}/events") as sse:
            lines = [ln for ln in sse.iter_lines() if ln.startswith("data: ")]
        assert json.loads(lines[-1][6:])["status"] == "done"
        work = client.get(f"/bowlers/{bowler['id']}/workload").json()
        assert sum(p["balls"] for p in work["points"]) == 3
        assert len(client.get("/bowlers").json()) == writes + 1
    for log in logs:
        text = log.read_text(encoding="utf-8")
        assert "database is locked" not in text, text[-2000:]
        assert "OfflineViolation" not in text, text[-2000:]
