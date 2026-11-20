"""Local job worker: polls the ``jobs`` table, runs the pipeline, stores the results.

Run it with ``python -m bowlsense.worker``. Jobs are claimed atomically (``BEGIN IMMEDIATE`` +
``UPDATE ... RETURNING``), progress is written to the job row for the SSE stream, and failures
are retried up to ``jobs.max_attempts`` times with the traceback kept in ``last_error``.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from bowlsense.config import Settings, load_settings
from bowlsense.db.repo import JobRecord, Repository
from bowlsense.db.session import get_engine
from bowlsense.errors import BowlSenseError, NotFoundError
from bowlsense.local_settings import LocalSettings, effective_settings, load_local_settings
from bowlsense.offline import enable_if_requested
from bowlsense.pipeline import PipelineOptions, run_pipeline
from bowlsense.schemas import SessionResult
from bowlsense.storage import OUTPUTS_DIR, LocalStorage

log = logging.getLogger(__name__)
POSE_SUFFIX = ".pose.json"


def pose_upload_path(video_path: str) -> str:
    """Storage path of the optional keypoints file uploaded with a video."""
    return video_path.rsplit(".", 1)[0] + POSE_SUFFIX


class Worker:
    """Runs queued jobs one at a time."""

    def __init__(self, settings: Settings, repo: Repository, storage: LocalStorage) -> None:
        """Bind to the database and storage."""
        self.settings, self.repo, self.storage = settings, repo, storage

    def options_for(self, job: JobRecord, local: LocalSettings) -> PipelineOptions:
        """Pipeline options from the session, its bowler profile and the local settings."""
        session = self.repo.get_session(job.session_id)
        opts = dict(session.options or {})
        bowling_arm = opts.get("bowling_arm")
        history = None
        recorded = (session.recorded_at or session.created_at).date()
        if session.bowler_id is not None:
            bowler = self.repo.get_bowler(session.bowler_id)
            bowling_arm = bowling_arm or bowler.bowling_arm
            history = self.repo.bowler_ball_history(session.bowler_id, session.id)
        out = self.storage.session_output_dir(session.id)
        uploaded_pose = self.storage.resolve(pose_upload_path(session.video_path))
        cached_pose = out / "pose.json"
        pose_json: Path | None = None
        if uploaded_pose.is_file():
            pose_json = uploaded_pose
        elif opts.get("reuse_pose") and cached_pose.is_file():
            pose_json = cached_pose  # calibration re-run: skip pose estimation
        return PipelineOptions(
            backend=local.pose_backend,
            allow_low_fps=bool(opts.get("allow_low_fps", False)),
            bowler_track_id=opts.get("bowler_track_id"),
            pose_json=pose_json,
            bowling_arm=bowling_arm,
            calibration_points=session.calibration,
            use_llm=local.use_llm,
            slow_motion=opts.get("slow_motion"),
            camera_view=opts.get("camera_view"),
            recorded_on=recorded,
            workload_history=history,
        )

    def process(self, job: JobRecord) -> SessionResult | None:
        """Run one claimed job; returns the result, or ``None`` if it failed."""
        session = self.repo.get_session(job.session_id)
        last = [0.0, -1.0]  # time and percent of the last progress write

        def progress(pct: float, stage: str) -> None:
            now = time.monotonic()
            if pct >= 100 or pct - last[1] >= 1 or now - last[0] > 1:
                self.repo.update_progress(job.id, pct, stage)
                last[:] = [now, pct]

        try:
            local = load_local_settings(self.settings)
            settings = effective_settings(self.settings, local)
            out = self.storage.session_output_dir(session.id)
            options = self.options_for(job, local)
            result = run_pipeline(
                self.storage.resolve(session.video_path),
                out,
                settings,
                session_id=session.id,
                options=options,
                progress=progress,
            )
            rel = f"{OUTPUTS_DIR}/{session.id}"
            self.repo.save_results(
                session.id,
                result,
                pdf_path=f"{rel}/{result.artifacts['report']}",
                overlay_path=f"{rel}/{result.artifacts['overlay']}",
            )
            self.repo.complete_job(job.id)
            log.info("job %s done (%d deliveries)", job.id, len(result.deliveries))
            return result
        except Exception as exc:
            if isinstance(exc, NotFoundError) and self._deleted(job):
                log.info("job %s: its session was deleted while running; skipped", job.id)
                return None
            detail = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            log.error("job %s failed: %s", job.id, exc)
            # Input problems (bad video, no bowler, poor pose) fail the same way every time:
            # don't retry them. Unexpected errors get the configured retries.
            attempts = job.attempts if isinstance(exc, BowlSenseError) else None
            try:
                self.repo.fail_job(
                    job.id, detail, attempts if attempts else self.settings.jobs.max_attempts
                )
            except NotFoundError:
                log.info("job %s: its session was deleted while running; skipped", job.id)
            return None

    def _deleted(self, job: JobRecord) -> bool:
        """True if the job (and so its session) no longer exists."""
        try:
            self.repo.get_job(job.id)
        except NotFoundError:
            return True
        return False

    def run_once(self) -> bool:
        """Claim and run the next queued job; ``False`` if there was none."""
        job = self.repo.claim_next_job()
        if job is None:
            return False
        log.info("job %s claimed (attempt %d) at %s", job.id, job.attempts, datetime.now(UTC))
        self.process(job)
        return True

    def run_forever(self, stop: threading.Event | None = None) -> None:
        """Poll every ``jobs.poll_interval_s`` until ``stop`` is set."""
        stop = stop or threading.Event()
        log.info("worker polling every %.1f s", self.settings.jobs.poll_interval_s)
        while not stop.is_set():
            try:
                busy = self.run_once()
            except Exception:  # one bad job or a DB hiccup must never stop the worker
                log.exception("worker: unexpected error; continuing")
                busy = False
            if not busy:
                stop.wait(self.settings.jobs.poll_interval_s)


def build_worker(settings: Settings | None = None) -> Worker:
    """A worker on the configured database and storage."""
    settings = settings or load_settings()
    engine = get_engine(str(settings.storage.db_file))
    return Worker(settings, Repository(engine), LocalStorage(settings.storage.root_path))


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m bowlsense.worker [--once]``."""
    parser = argparse.ArgumentParser(prog="bowlsense-worker", description=__doc__)
    parser.add_argument("--once", action="store_true", help="run at most one job, then exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(message)s")
    enable_if_requested()
    worker = build_worker()
    if args.once:
        worker.run_once()
    else:
        try:
            worker.run_forever()
        except KeyboardInterrupt:
            log.info("worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
