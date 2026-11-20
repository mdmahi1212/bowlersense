"""BowlSense HTTP API (FastAPI), bound to 127.0.0.1 only.

Run with ``python -m bowlsense.api`` (or via ``run.py``). Analysis runs in the separate worker
process (``bowlsense.worker``); this app stores uploads, queues jobs and serves results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from bowlsense import __version__
from bowlsense.analysis.compare import ReferenceAction, load_references
from bowlsense.analysis.fatigue import assess_workload
from bowlsense.coach.drills import DRILLS, Drill
from bowlsense.coach.llm import ChatTurn, OllamaClient, OllamaUnavailableError, coach_reply
from bowlsense.coach.prompts import CoachSummary
from bowlsense.config import Settings, load_settings
from bowlsense.db.repo import BowlerRecord, JobRecord, Repository, SessionRecord
from bowlsense.db.session import get_engine
from bowlsense.device import gpu_status
from bowlsense.errors import (
    BowlSenseError,
    CalibrationError,
    ModelNotFoundError,
    NotFoundError,
    UnsupportedMediaError,
)
from bowlsense.geometry.pitch import calibrate
from bowlsense.image_mode import ImageAnalysis, ImageCheck, analyze_image
from bowlsense.local_settings import (
    LocalSettings,
    effective_settings,
    load_local_settings,
    save_local_settings,
)
from bowlsense.schemas import SKELETON_EDGES, DeliveryMetrics, SessionResult, WorkloadResult
from bowlsense.storage import OUTPUTS_DIR, LocalStorage, PathTraversalError, UploadTooLargeError
from bowlsense.worker import pose_upload_path

log = logging.getLogger(__name__)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --- request / response bodies ------------------------------------------------------------------


class BowlerIn(BaseModel):
    """New bowler profile."""

    name: str = Field(min_length=1, max_length=200)
    height_cm: float | None = Field(default=None, gt=50, lt=260)
    bowling_arm: Literal["right", "left"] | None = None
    bowling_type: str | None = Field(default=None, max_length=50)
    dob: date | None = None


class SessionOut(BaseModel):
    """A session with its latest job and (when done) its result."""

    session: SessionRecord
    job: JobRecord | None
    result: SessionResult | None = None
    delivery_ids: list[str] = Field(default_factory=list)


class CalibrationIn(BaseModel):
    """Four crease corners in image pixels (see ``geometry.pitch`` for the order)."""

    points: list[list[float]] = Field(min_length=4, max_length=4)
    rerun: bool = True


class CoachIn(BaseModel):
    """A chat message to the coach."""

    message: str = Field(min_length=1, max_length=2000)


class CoachOut(BaseModel):
    """The coach's reply."""

    reply: str
    source: Literal["ollama", "template"]
    note: str = ""


class ConfirmIn(BaseModel):
    """Body of ``DELETE /data``."""

    confirm: str


class DeliveryOut(BaseModel):
    """One delivery with what the detail page needs from its session."""

    delivery_id: str
    session_id: str
    fps: float
    width: int
    height: int
    bowling_arm: Literal["right", "left"]
    view: str
    overlay: str  # storage path of the session's overlay video
    pose: str  # storage path of the session's pose.json
    delivery_count: int
    delivery: DeliveryMetrics


class ReferencesOut(BaseModel):
    """Reference actions for the compare page."""

    references: list[ReferenceAction]
    edges: list[list[str]]


class SettingsOut(BaseModel):
    """Local settings plus read-only facts for the settings page."""

    settings: LocalSettings
    storage_bytes: int
    gpu: dict[str, bool]


class WorkloadPoint(BaseModel):
    """ACWR on one day with a session."""

    day: date
    balls: int
    acwr: float | None
    flagged: bool


class WorkloadOut(BaseModel):
    """A bowler's workload history."""

    bowler_id: str
    points: list[WorkloadPoint]
    safe_band: tuple[float, float] = (0.8, 1.3)
    flag_above: float
    latest: WorkloadResult | None


# --- app ----------------------------------------------------------------------------------------


class AppState:
    """Shared objects for the request handlers."""

    def __init__(self, settings: Settings) -> None:
        """Open the (migrated) database and the storage root."""
        self.settings = settings
        self.repo = Repository(get_engine(str(settings.storage.db_file)))
        self.storage = LocalStorage(settings.storage.root_path)

    def output_dir(self, session_id: str) -> Path:
        """``outputs/<session_id>`` (absolute)."""
        return self.storage.resolve(f"{OUTPUTS_DIR}/{session_id}", within=OUTPUTS_DIR)

    def result(self, session_id: str) -> SessionResult | None:
        """The stored ``SessionResult`` with artifact paths made storage-relative."""
        path = self.output_dir(session_id) / "result.json"
        if not path.is_file():
            return None
        result = SessionResult.model_validate_json(path.read_text(encoding="utf-8"))
        prefix = f"{OUTPUTS_DIR}/{session_id}/"
        return result.model_copy(
            update={
                "artifacts": {k: prefix + v for k, v in result.artifacts.items()},
                "deliveries": [
                    d.model_copy(
                        update={"keyframes": {e: prefix + p for e, p in d.keyframes.items()}}
                    )
                    for d in result.deliveries
                ],
            }
        )

    def coach_summary(self, session_id: str) -> CoachSummary:
        """The metrics summary the coach chat uses."""
        path = self.output_dir(session_id) / "coach_summary.json"
        if not path.is_file():
            raise HTTPException(409, "The session has no results yet; the coach needs them.")
        return CoachSummary.model_validate_json(path.read_text(encoding="utf-8"))


def _error(status: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": str(exc)})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API for ``settings`` (default: the config file)."""
    settings = settings or load_settings()
    state = AppState(settings)
    app = FastAPI(title="BowlSense", version=__version__, docs_url="/docs", redoc_url=None)
    app.state.bowlsense = state
    web = int(os.environ.get("BOWLSENSE_WEB_PORT", settings.server.web_port))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"http://localhost:{web}", f"http://127.0.0.1:{web}"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, exc)

    @app.exception_handler(UnsupportedMediaError)
    async def _unsupported(_: Request, exc: UnsupportedMediaError) -> JSONResponse:
        return _error(415, exc)

    @app.exception_handler(UploadTooLargeError)
    async def _too_large(_: Request, exc: UploadTooLargeError) -> JSONResponse:
        return _error(413, exc)

    @app.exception_handler(PathTraversalError)
    async def _traversal(_: Request, exc: PathTraversalError) -> JSONResponse:
        return _error(400, exc)

    @app.exception_handler(ModelNotFoundError)
    async def _no_model(_: Request, exc: ModelNotFoundError) -> JSONResponse:
        return _error(503, exc)

    @app.exception_handler(BowlSenseError)
    async def _bowlsense(_: Request, exc: BowlSenseError) -> JSONResponse:
        return _error(422, exc)  # bad input, no bowler found, poor pose quality, ...

    # --- health / settings ----------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Model files present, GPU availability, version."""
        models = settings.storage.models_path
        wanted = {
            "yolo": settings.detect.yolo_model,
            "pose_rtm": settings.pose.rtm_model,
            "pose_mediapipe": settings.pose.mediapipe_model,
            "phase_model": settings.phases.model_file,
            "delivery_type": settings.delivery.type_model_file,
        }
        return {
            "status": "ok",
            "version": __version__,
            "models": {k: (models / v).is_file() for k, v in wanted.items()},
            "gpu": gpu_status(),
        }

    @app.get("/settings")
    def get_settings() -> SettingsOut:
        """Local settings, storage usage and GPU status."""
        return SettingsOut(
            settings=load_local_settings(settings),
            storage_bytes=state.storage.usage_bytes(),
            gpu=gpu_status(),
        )

    @app.put("/settings")
    def put_settings(body: LocalSettings) -> SettingsOut:
        """Save local settings (applied to jobs started afterwards)."""
        save_local_settings(settings, body)
        return get_settings()

    @app.get("/llm/models")
    def llm_models() -> dict[str, Any]:
        """Models pulled into the local Ollama (``available: false`` if it isn't running)."""
        cfg = effective_settings(settings, load_local_settings(settings)).coach
        client = OllamaClient(cfg)
        try:
            return {"available": True, "models": client.models(), "selected": cfg.model}
        except OllamaUnavailableError as exc:
            return {"available": False, "models": [], "selected": cfg.model, "detail": str(exc)}
        finally:
            client.close()

    @app.delete("/data")
    def delete_data(body: ConfirmIn) -> dict[str, str]:
        """Delete every local row and file. Requires ``{"confirm": "DELETE"}``."""
        if body.confirm != "DELETE":
            raise HTTPException(400, 'Send {"confirm": "DELETE"} to delete all data.')
        state.repo.delete_all_data(state.storage)
        return {"status": "deleted"}

    # --- bowlers --------------------------------------------------------------------------

    @app.get("/bowlers")
    def list_bowlers() -> list[BowlerRecord]:
        """All bowler profiles."""
        return state.repo.list_bowlers()

    @app.post("/bowlers", status_code=201)
    def create_bowler(body: BowlerIn) -> BowlerRecord:
        """Create a bowler profile."""
        return state.repo.create_bowler(
            body.name, height_cm=body.height_cm, bowling_arm=body.bowling_arm,
            bowling_type=body.bowling_type, dob=body.dob.isoformat() if body.dob else None,
        )  # fmt: skip

    @app.get("/bowlers/{bowler_id}/workload")
    def bowler_workload(bowler_id: str) -> WorkloadOut:
        """ACWR on each day the bowler has a finished session."""
        history = state.repo.bowler_ball_history(bowler_id)
        by_day: dict[date, int] = {}
        for day, balls in history:
            by_day[day] = by_day.get(day, 0) + balls
        points, latest = [], None
        for day, balls in sorted(by_day.items()):
            earlier = [(d, b) for d, b in by_day.items() if d < day]
            latest = assess_workload(earlier, day, balls, settings.workload)
            points.append(
                WorkloadPoint(day=day, balls=balls, acwr=latest.acwr, flagged=latest.flagged)
            )
        return WorkloadOut(
            bowler_id=bowler_id,
            points=points,
            flag_above=settings.workload.acwr_flag,
            latest=latest,
        )

    # --- sessions -------------------------------------------------------------------------

    def session_out(session_id: str) -> SessionOut:
        session = state.repo.get_session(session_id)
        result = state.result(session_id) if session.status == "done" else None
        ids = [d.id for d in state.repo.list_deliveries(session_id)] if result else []
        return SessionOut(
            session=session, job=state.repo.latest_job_for_session(session_id),
            result=result, delivery_ids=ids,
        )  # fmt: skip

    @app.post("/sessions", status_code=201)
    def create_session(
        file: Annotated[UploadFile, File(description="video: mp4, mov or avi")],
        bowler_id: Annotated[str | None, Form()] = None,
        bowling_arm: Annotated[Literal["right", "left"] | None, Form()] = None,
        allow_low_fps: Annotated[bool, Form()] = False,
        bowler_track_id: Annotated[int | None, Form()] = None,
        recorded_at: Annotated[datetime | None, Form()] = None,
        pose_file: Annotated[
            UploadFile | None, File(description="optional PoseSequence JSON")
        ] = None,
    ) -> SessionOut:
        """Store the upload as ``raw/<uuid>.<ext>`` (streamed in chunks) and queue analysis."""
        name = file.filename or ""
        suffix = Path(name).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            raise HTTPException(415, "Images are analysed with POST /analyze/image.")
        if bowler_id:
            state.repo.get_bowler(bowler_id)
        max_bytes = settings.api.max_upload_mb * 1024 * 1024
        rel = state.storage.save_upload(file.file, name, VIDEO_EXTENSIONS, max_bytes)
        try:
            if pose_file is not None and pose_file.filename:
                pose_rel = pose_upload_path(rel)
                target = state.storage.resolve(pose_rel)
                saved = state.storage.save_upload(
                    pose_file.file, "pose.json", (".json",), max_bytes
                )
                state.storage.resolve(saved).replace(target)
            options = {"allow_low_fps": allow_low_fps, "bowler_track_id": bowler_track_id,
                       "bowling_arm": bowling_arm}  # fmt: skip
            session = state.repo.create_session(
                rel, bowler_id=bowler_id or None, recorded_at=recorded_at,
                options=options, original_name=name,
            )  # fmt: skip
        except BaseException:
            state.storage.resolve(rel).unlink(missing_ok=True)
            state.storage.resolve(pose_upload_path(rel)).unlink(missing_ok=True)
            raise
        state.repo.create_job(session.id)
        return session_out(session.id)

    @app.get("/sessions")
    def list_sessions(bowler_id: Annotated[str | None, Query()] = None) -> list[SessionRecord]:
        """Sessions, newest first (optionally one bowler's)."""
        return state.repo.list_sessions(bowler_id)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> SessionOut:
        """Status, latest job, and the ``SessionResult`` when done."""
        return session_out(session_id)

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, str]:
        """Delete a session, its rows and its files."""
        session = state.repo.get_session(session_id)
        state.repo.delete_session(session_id, state.storage)
        state.storage.resolve(pose_upload_path(session.video_path)).unlink(missing_ok=True)
        return {"status": "deleted"}

    @app.get("/sessions/{session_id}/events")
    async def session_events(request: Request, session_id: str) -> StreamingResponse:
        """Server-Sent Events: ``status``, ``progress`` (0-100), ``stage``; ends when finished."""
        state.repo.get_session(session_id)

        async def stream() -> AsyncIterator[str]:
            last: dict[str, Any] | None = None
            deadline = datetime.now(UTC) + timedelta(hours=6)
            while datetime.now(UTC) < deadline:
                if await request.is_disconnected():
                    return
                try:
                    session = state.repo.get_session(session_id)
                except NotFoundError:
                    yield f"event: deleted\ndata: {json.dumps({'status': 'deleted'})}\n\n"
                    return
                job = state.repo.latest_job_for_session(session_id)
                event = {
                    "status": session.status,
                    "progress": job.progress if job else 0.0,
                    "stage": job.stage if job else None,
                    "attempts": job.attempts if job else 0,
                    "error": session.error,
                }
                if event != last:
                    yield f"data: {json.dumps(event)}\n\n"
                    last = event
                if session.status in ("done", "failed"):
                    return
                await asyncio.sleep(0.5)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    @app.post("/sessions/{session_id}/calibration")
    def save_calibration(session_id: str, body: CalibrationIn) -> SessionOut:
        """Save 4 crease corners; re-run the analysis (reusing the stored pose) if requested."""
        try:
            calibrate([(p[0], p[1]) for p in body.points])
        except (CalibrationError, IndexError) as exc:
            raise HTTPException(422, f"Invalid calibration: {exc}") from exc
        session = state.repo.save_calibration(session_id, body.points)
        if body.rerun and session.status in ("done", "failed"):
            state.repo.set_session_options(
                session_id, {**(session.options or {}), "reuse_pose": True}
            )
            state.repo.set_session_status(session_id, "queued")
            state.repo.create_job(session_id)
        return session_out(session_id)

    @app.get("/sessions/{session_id}/report.pdf")
    def session_pdf(session_id: str) -> FileResponse:
        """The PDF report."""
        state.repo.get_session(session_id)
        pdf = state.output_dir(session_id) / "report.pdf"
        if not pdf.is_file():
            raise HTTPException(404, "The report isn't ready yet.")
        return FileResponse(
            pdf, media_type="application/pdf", filename=f"bowlsense_{session_id[:8]}.pdf"
        )

    @app.post("/sessions/{session_id}/coach")
    def coach_chat(session_id: str, body: CoachIn) -> CoachOut:
        """Chat with the coach about this session (history kept in the database)."""
        state.repo.get_session(session_id)
        summary = state.coach_summary(session_id)
        history = [
            ChatTurn(role=m.role, content=m.content)
            for m in state.repo.list_coach_messages(session_id)
        ]
        cfg = effective_settings(settings, load_local_settings(settings)).coach
        reply = coach_reply(summary, history, body.message, cfg)
        state.repo.add_coach_message(session_id, "user", body.message)
        state.repo.add_coach_message(session_id, "assistant", reply.text)
        return CoachOut(reply=reply.text, source=reply.source, note=reply.note)

    @app.get("/sessions/{session_id}/coach")
    def coach_history(session_id: str) -> list[ChatTurn]:
        """The chat so far."""
        return [
            ChatTurn(role=m.role, content=m.content)
            for m in state.repo.list_coach_messages(session_id)
        ]

    # --- deliveries / files ---------------------------------------------------------------

    @app.get("/deliveries/{delivery_id}")
    def get_delivery(delivery_id: str) -> DeliveryOut:
        """Full metrics of one delivery, plus its session's video facts."""
        ref = state.repo.get_delivery(delivery_id)
        result = state.result(ref.session_id)
        if result is None or ref.idx >= len(result.deliveries):
            raise HTTPException(404, "The delivery's results are missing.")
        return DeliveryOut(
            delivery_id=delivery_id, session_id=ref.session_id, fps=result.fps,
            width=result.width, height=result.height, bowling_arm=result.bowling_arm,
            view=result.view, overlay=result.artifacts["overlay"], pose=result.artifacts["pose"],
            delivery_count=len(result.deliveries), delivery=result.deliveries[ref.idx],
        )  # fmt: skip

    @app.post("/analyze/image")
    def analyze_image_endpoint(
        file: Annotated[UploadFile, File(description="photo: jpg or png")],
        phase: Annotated[Literal["auto", "bfc", "ffc", "release"], Form()] = "auto",
        view: Annotated[Literal["auto", "side", "front"], Form()] = "auto",
        bowling_arm: Annotated[Literal["right", "left"] | None, Form()] = None,
    ) -> ImageAnalysis:
        """Image mode (synchronous): pose on one photo and the single-frame posture checks."""
        max_bytes = settings.api.max_upload_mb * 1024 * 1024
        rel = state.storage.save_upload(file.file, file.filename or "", IMAGE_EXTENSIONS, max_bytes)
        photo = state.storage.resolve(rel)
        image_id = photo.stem
        out = state.storage.resolve(f"{OUTPUTS_DIR}/images/{image_id}", within=OUTPUTS_DIR)
        try:
            effective = effective_settings(settings, load_local_settings(settings))
            result = analyze_image(
                photo, effective, phase=phase, view=view, bowling_arm=bowling_arm,
                annotated_out=out / "annotated.png",
            )  # fmt: skip
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        finally:
            photo.unlink(missing_ok=True)  # only the annotated copy is kept
        return result.model_copy(
            update={"annotated": f"{OUTPUTS_DIR}/images/{image_id}/annotated.png"}
        )

    @app.get("/sessions/{session_id}/video")
    def session_video(session_id: str) -> FileResponse:
        """The uploaded video (HTTP Range supported), for the overlay toggle."""
        session = state.repo.get_session(session_id)
        video = state.storage.resolve(session.video_path, within="raw")
        if not video.is_file():
            raise HTTPException(404, "The uploaded video is missing.")
        return FileResponse(video)

    @app.get("/drills")
    def drills() -> dict[str, list[Drill]]:
        """The drill library: fault code -> drills."""
        return {code: list(items) for code, items in DRILLS.items()}

    @app.get("/references")
    def references() -> ReferencesOut:
        """The shipped reference actions (synthetic, never real players) and skeleton edges."""
        refs = load_references(settings.compare.reference_path)
        return ReferencesOut(references=refs, edges=[list(e) for e in SKELETON_EDGES])

    @app.get("/files/{path:path}")
    def get_file(path: str) -> FileResponse:
        """Serve a file from ``outputs/`` only (HTTP Range supported for video seeking)."""
        target = state.storage.resolve(path, within=OUTPUTS_DIR)
        if not target.is_file():
            raise HTTPException(404, "File not found.")
        return FileResponse(target)

    return app


def main() -> None:
    """Run the API with uvicorn on ``server.host:server.api_port`` (loopback only)."""
    import uvicorn

    from bowlsense.offline import enable_if_requested

    logging.basicConfig(level=logging.INFO, format="%(asctime)s api %(message)s")
    enable_if_requested()
    settings = load_settings()
    port = int(os.environ.get("BOWLSENSE_API_PORT", settings.server.api_port))
    uvicorn.run(create_app(settings), host=settings.server.host, port=port, log_level="info")


if __name__ == "__main__":
    main()


API_MODELS: tuple[type[BaseModel], ...] = (
    BowlerIn, BowlerRecord, SessionRecord, JobRecord, SessionOut, CalibrationIn, CoachIn,
    CoachOut, ChatTurn, ConfirmIn, SettingsOut, LocalSettings, WorkloadPoint, WorkloadOut,
    ReferencesOut, ReferenceAction, Drill, DeliveryOut, ImageAnalysis, ImageCheck,
)  # fmt: skip
"""Request/response bodies exported (with ``schemas.EXPORTED_MODELS``) for the web types."""
