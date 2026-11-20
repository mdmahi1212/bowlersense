# BowlSense — Progress

## Phase 14: Next.js web app ✅ (2026-11-20)

- **Stack**: Next.js 16.4.6 (App Router, TypeScript, Turbopack), React 19.3, Tailwind CSS 4, Recharts 3, all pinned exactly.
  - Local only: no Google fonts (system font stack), no telemetry (`NEXT_TELEMETRY_DISABLED`), and dev/start bound to 127.0.0.1.
  - The API address is read per request from `BOWLSENSE_API_URL` (accepted only if it's a loopback URL) and handed to the browser in a `<meta>` tag, so `run.py` can move ports without a rebuild.
  - UI primitives (Button, Card, Badge, states) follow shadcn/ui's copy-in pattern (`components/ui/`) without its CLI or registry download.
- **Types are generated, never hand-written.** `npm run gen-types` runs `python -m bowlsense.schemas --with-api` (contracts plus API bodies, serialization mode) and then json-schema-to-typescript, producing `lib/types.ts`. `lib/api.ts` is the typed client, including an XHR upload with progress.
- **API additions for the UI**: `GET /sessions/{id}/video` (raw upload, Range), `/drills`, `/references`, and `/deliveries/{id}`, which now returns `DeliveryOut` (the delivery plus session fps, size, arm, view, overlay and pose paths).
- **Pages**:
  - **Upload** (`/`): drag and drop, bowler picker or inline create, low-fps opt-in, optional keypoints JSON, recording tips, upload progress bar.
  - **Sessions** (`/sessions`): list with filter by bowler and delete.
  - **Session dashboard** (`/sessions/[id]`): live status via SSE with a 2 s polling fallback. When done it shows:
    - technique score, injury gauge (band in words), legality badge with the screening note, consistency;
    - the delivery table, fatigue trend, and pitch map with the click-4-corners calibration UI (saves, then re-runs);
    - the coach report, PDF download, coach chat, and closest references.
  - **Delivery detail** (`/deliveries/[id]`):
    - player with overlay/raw toggle, ← → frame stepping, phase markers and jump buttons;
    - keyframe strip (click to seek);
    - elbow, front-knee and trunk charts synced to the player cursor, click to seek, with the 15° band on the elbow chart;
    - fault list with severity, "jump to frame" and drills, plus checks not possible from this view;
    - body risk map and key numbers.
  - **Compare** (`/compare`): pick a delivery and a reference; synced side-by-side skeleton playback (time-normalised from BFC to follow-through, play/scrub, mirror for left-armers) and per-joint deviation bars.
  - **Bowlers** (`/bowlers`): profiles, create, session history, ACWR chart with the 0.8–1.3 band and the 1.5 flag line.
  - **Settings** (`/settings`): Ollama model picker (from `/llm/models`), pose backend, GPU on/off, LLM on/off, model files present, storage usage, and "Delete all my data" (type DELETE).
  - Every page handles loading, empty and error states. `useAsync` turns rejections into text, so there are no unhandled promise rejections. Risk colours always come with text labels. The theme is dark-first.
- **Tests**:
  - `npm run lint` (ESLint and Prettier) is clean and `npm run build` passes.
  - Vitest: 10 unit tests (utils, API client errors and URLs, runtime API address).
  - Playwright smoke test (`npm run test:e2e`, using installed Edge on Windows, so no browser download): `tests/e2e/stack.mjs` starts a throwaway API and worker with the offline guard on and a pre-migrated DB. The browser uploads a rendered synthetic delivery with its keypoints, waits for the result, then checks the technique score, the flagged elbow, the coach report and the delivery detail page (knee-collapse fault, angle charts, video), and visits every page with no page errors.
  - `scripts/check.py` now runs lint, build, unit and e2e.

## Phase 13: SQLite schema + Alembic + local storage ✅ (2026-11-20)

Most of this was built in Phase 1: ORM tables, Alembic, WAL, foreign keys, cascades, `BEGIN IMMEDIATE` claims, safe storage paths, and the repo-level two-process test. Phase 12 added migration `0002`. This phase adds the gate's real-process check:
- `test_processes.py`: the actual `python -m bowlsense.api` (uvicorn, on a free port) and `python -m bowlsense.worker` run as two processes on one temporary SQLite file with `BOWLSENSE_OFFLINE=1`. Three uploads are processed while the API keeps writing bowler rows. Every session finishes, SSE reports done, workload sums the balls, and neither log contains `database is locked` or an offline violation.
- `test_api.py`: tests for the raw-video endpoint (Range), `/drills` and `/references`.
- Gate status: `alembic upgrade head` on an empty DB (`test_db.py`), cascade delete of rows and files (`test_db.py`, `test_api.py`), and concurrent API and worker writes (`test_db.py`, `test_processes.py`) all pass.

## Phase 12: Full pipeline + CLI + FastAPI + worker ✅ (2026-11-20)

- **`pipeline.run_pipeline`** runs a video through every stage and writes overlay.mp4, keyframes, charts, report.pdf, pose.json, coach_summary.json and result.json. It returns a `SessionResult`, reports progress (0–100 % plus a stage name) through a callback, and works with the offline guard on.
  - `DeliveryMetrics` gained: clip offsets, alignment, fault measurements and not-measured reasons, delivery type, ball notes and track, reference comparisons, and keyframe paths. `stride_pct` and `release_height_pct` are `None` when the view can't measure them.
  - `SessionResult` gained: video facts, view, arm, pose backend, the detailed consistency, and the coach source and note. `consistency_score` is `None` with one delivery instead of a fake number.
- **CLI** is now a thin wrapper over `run_pipeline`. Its `result.json` *is* the `SessionResult`.
- **`worker.py`** (`python -m bowlsense.worker [--once]`):
  - polls `jobs`, claims atomically, and throttles progress writes for the SSE stream;
  - builds options from the session, the bowler profile (arm and workload history) and the local settings;
  - stores results in the DB (deliveries, angle series, faults, summary);
  - retries up to `jobs.max_attempts` and keeps the traceback in `last_error`.
  - A calibration re-run reuses the stored pose.json, so pose estimation isn't repeated.
- **`api.py`** (`python -m bowlsense.api`, 127.0.0.1 only; CORS only for the local web port) covers every Section 9 endpoint except `/analyze/image` (Phase 15):
  - health (model files, GPU), settings get/put, `/llm/models`, `DELETE /data` (needs `{"confirm": "DELETE"}`);
  - bowlers, `/bowlers/{id}/workload` (ACWR per day plus the 0.8–1.3 band);
  - sessions: create, list, get, delete, SSE events, calibration (validated, then re-queued), report.pdf, coach chat with history;
  - `/deliveries/{id}`, and `/files/{path}` serving `outputs/` only, with HTTP Range (206) and traversal rejected.
  - Uploads stream to `raw/<uuid>.<ext>`: 415 for other types (images point to `/analyze/image`), 413 above the size limit, and partial files are removed. An optional `pose_file` (PoseSequence JSON) can be uploaded with a video to skip the pose model; the tests and the Playwright smoke test use it.
- **Supporting changes**:
  - `local_settings.py`: settings-page overrides (Ollama model, pose backend, GPU on/off, LLM on/off) in `data/settings.json`;
  - `device.py`: the GPU switch, honoured by RTMPose and YOLO;
  - Alembic `0002`: `sessions.options` and `original_name`;
  - repo: `save_results`, `list_deliveries`, `get_delivery`, `set_session_options`;
  - `synth/sessions.py`: a multi-ball session generator;
  - FastAPI, Starlette, Uvicorn and python-multipart pinned.
- **Tests**:
  - `test_pipeline_synthetic.py` (5): a 3-ball video run with the offline guard on, checking phases, legality, faults, angles, references, workload with history, artifacts, progress, and the no-profile/one-ball path.
  - `test_api.py` (18): upload → worker → results, SSE, Range, traversal, PDF, coach chat, workload, calibration re-run, retry → failed with traceback, cascade delete of rows and files, delete-all confirmation, validation, CORS.

## Phase 11: Coach (drills, local Ollama + template fallback, chat) ✅ (2026-11-20)

- `coach/drills.py`: 2–3 drills (name, description, reps) for each of the 10 fault codes.
- `coach/prompts.py`: `summarise_session` builds the only thing the coach sees, a JSON summary of the metrics (never video or images). It ranks issues by weighted severity and keeps the top 3, and also covers legality, injury band and drivers, ball speeds, fatigue alerts, consistency and the closest reference. Also holds the system prompt (friendly, concrete, cites numbers, no medical diagnosis, ≤ 250 words, top 3 issues) and the chat context.
- `coach/llm.py`:
  - `OllamaClient` talks only to the configured localhost URL. It checks `GET /api/tags` for the model, then calls `POST /api/chat` with `stream: false`, a 120 s timeout and 1 retry.
  - `generate_report` uses Ollama and trims replies to the word limit. It falls back to the template report on any failure (Ollama stopped, model not pulled, timeouts, bad payloads) and records why in `note`.
  - `template_report` stands on its own: intro, top issues with their numbers and drills, elbow screening, injury screening, speed, fatigue, consistency and disclaimers.
  - `coach_reply` handles chat with the session metrics plus the last N exchanges as context. Without Ollama, it gives an offline answer matched to the question.
- CLI: `--no-llm` added. `result.json` gains `coach_summary` and `coach_report`, and the PDF includes the coach's notes.
- Tests: `test_coach.py` (19 tests) mocks Ollama with respx (running, stopped, model missing, retry then fail, retry then recover, word trimming, chat history). The existing CLI tests pass `--no-llm`. `respx==0.23.1` is pinned in requirements-dev.
- Known: Ollama itself isn't installed on this machine yet, so the real-LLM path is verified against mocks only.

## Phase 10: Module G (comparison, consistency, bowler ID) ✅ (2026-11-20)

**Done**
- `analysis/compare.py`: per-frame joint angles (bowling elbow, front knee, bowling shoulder, trunk lean, shoulder-vs-hip line tilt; deg/180), time-normalised from BFC to follow-through end (50 points). DTW via tslearn → `similarity % = 100·exp(−dtw/2.0)` and per-joint mean deviation (deg) along the DTW path. On synthetic data an identical action is 100%, a close one ~80%, and a distorted one (22° elbow + 30° knee collapse) ~63%. Invariant to fps (≥ 90%), bowling arm (> 99%, joints by role) and camera distance (> 99%).
- `scripts/make_references.py` → `data/reference/{side_on,semi_open,front_on}.json` ("Reference: textbook …", synthetic; never a real player). Each delivery now carries `comparisons` (most similar first).
- **Consistency** = 100 − mean coefficient of variation of technique score, elbow extension, stride, release height and knee collapse (a 10-unit floor keeps near-zero metrics from exploding). `None` with < 2 deliveries. Added to the session and `result.json`.
- `analysis/bowler_id.py`: a GRU over the angle curves; trains only with ≥ 3 bowlers × ≥ 20 deliveries, otherwise returns `"insufficient data"` with what's missing. (Feeding it from the database per bowler is part of Phase 12.)
- tslearn 0.10.0 pinned (scikit-learn is now a runtime dependency through it).

**Gate**: identical sequences → 100% similarity (and 0° deviation). 10 new tests (656 total). Bowler ID on 3 synthetic bowler styles × 20 deliveries → ≥ 80% on held-out deliveries; the insufficient-data path is tested.

**Note**: from a side-on camera, shoulder/hip alignment barely shows in 2D, so side-view DTW reflects overall joint-curve similarity rather than action type (the alignment module covers action type from front/behind views).

## Training on the real bowling dataset ✅ (2026-11-20)

**Dataset** (`C:\dev\dataset`, outside the repo; set `--dataset` or `BOWLSENSE_DATASET`): 40 club-cricket videos, 1280×720 at 30 fps, ~4.4 h. Split: 18 train / 8 test / 14 unlabelled challenge. Labels: **279 single-frame ball releases** (each preceded by an "is bowling" span), pace/spin, bowler hand, over/around the wicket; per-frame person boxes with roles; camera direction. Please confirm the dataset's licence allows use in your project.

**Pipeline** (`ml/train/`): `extract_poses` runs RTMPose on the dataset's own bowler boxes around every delivery (videos decoded sequentially, not by seeking, so frame indices match the labels exactly; run with the network blocked). Then `train_release`, `train_classifiers`, `train_delivery_type` (hand-crafted baseline) and `evaluate_pipeline`. Validation always holds out whole videos; the test split is used once. Reports go to `data/training/reports/`.

**Results on the held-out test split**
| task | trained model | baseline |
|---|---|---|
| Release frame (255 fixed 3 s crops; the release is randomly placed so the model can't learn the window shape) | **75% within ±1 frame, 86% ±2, 92% ±3, MAE 1.28 frames (43 ms)** | rule-based detector: 48% / 52% / 54%, MAE 4.87 frames |
| Bowling arm | **elbow height at the learned release: 94% (84.5% over all 278)**; synthetic gate 48/48 | spec's peak-wrist-speed rule: 59–62% |
| Pace vs spin (end to end: learned arm → learned release → classifier) | **BiGRU: 63.1% accuracy, 62.6% balanced** | majority 46.4% / 50%; hand-crafted logistic regression 47.6% (no better than the baseline, kept only as the report's baseline) |
| Camera view (existing rules) | 93% (front/behind 253/258; side 7/21, rare) | — |

**What this changed in BowlSense**
- `phases/model.py` (BiLSTM, release + per-frame phase heads) and `phases/seqclf.py` (BiGRU + attention) are used only when enabled / present. `phases.use_model: true` → release from the model, other events by rules around it (`release_source` records which).
- **Bowling arm**: the bowler profile / `--bowling-arm` wins; else elbow-at-learned-release (when the phase model is on); else the spec's wrist-speed rule. Why: on real 30 fps footage the wrist-speed rule fails (front/behind views hide the arm's swing; the front arm also goes high at the bound), and the wrist is a motion-blurred streak at release, while the elbow is reliable.
- Pace/spin is attached to each delivery when `models/delivery_type_seq.pt` exists.
- The trained weights (~840 KB) are committed (whitelisted in `.gitignore`), so a fresh clone works without the dataset.

**Tried and dropped (don't retry)**: a learned arm classifier with mirror augmentation (no better than chance on left-armers: 28 left-arm deliveries from only 3 bowlers); whole-window wrist-height rules (70%); comparing each arm "at its own release" (the front arm's own peak is at the bound; 0/24 synthetic).

**Known limits**: left-armers are underrepresented (3 bowlers), so arm accuracy for them is uncertain; pace/spin from distant 30 fps pose alone is modest. Default `phases.use_model` stays `false` as the spec says; **set it to `true` for real footage**.

## Phase 9: Module F (ball, calibration, speed, length, line) ✅ (2026-11-20)

**Done**
- `geometry/pitch.py`: 4 clicked popping-crease corners → homography to pitch metres (17.68 m between popping creases), with validation (count, repeated or collinear points, degenerate fit). **Airborne-ball fix**: a ground homography is wrong for a ball ~2 m up, so speed is measured in the vertical plane along the pitch centre line (`along_pitch`: the centre-line point at the same image column). This needs a side-on camera; otherwise speed is `None` with a reason. Auto-detecting crease lines (the spec's "or") isn't implemented; manual 4-point calibration is.
- `detect/ball.py`: YOLO "sports ball" (conf 0.1) in a 480 px window that follows the prediction, starting at the bowling hand at release. Kalman filter: constant velocity + constant vertical acceleration; gating; bridges ≤ 4 misses, never ends on a guess. **Bounce = the strongest down→up reversal of fitted image-y slopes** (a single-frame local max was fooled by 0.3 px noise).
- `analysis/delivery.py`: speed = slope of along-pitch distance over the first 5–10 detections × 3.6. Pitch position at the bounce (on the ground, so the homography is valid) → length zones (from the batter's stumps) and line bands (off side = bowler's left for a right-handed batter; configurable). **Length and line are reported only when the camera angle resolves them**: the homography's local derivatives give metres per pixel at the bounce. A low side-on camera sees the ground at a grazing angle (≈0.4 m sideways per px), so line is withheld there, and length from behind. Each withheld value comes with a note.
- Pipeline + CLI: `--calib calib.json` (`{"points": [[x, y] x 4]}`), `--bowling-arm`; ball metrics and track in `DeliveryAnalysis`; the PDF summary shows speed / length / line / pace-spin or why they're missing.

**Gate**: a synthetic 3D ball flight seen through a pinhole camera (perspective, airborne, gravity, bounce; 0.5 px noise; every 5th detection missing) gives speed **within ±5% in all 12 cases** (90/120/145 km/h × 120/240 fps × side-on and 11°-off cameras). No calibration → `None`; camera behind the bowler → `None` with a note. 35 tests.

**Not verified on real footage**: the dataset has no ball labels, so ball detection quality on real video is unmeasured.

## Phase 8: Rendering (overlay, keyframes, clipping, PDF) ✅ (2026-11-20)

**Installed**: imageio-ffmpeg 0.8.1 (bundles ffmpeg 9.0.2 with libx264, inside the venv; no system install), reportlab 5.1.0; dev: pypdf 7.1.1 (tests open and read the PDFs).

**Done**
- `render/ffmpeg.py`: `H264Writer` streams raw BGR frames into ffmpeg's libx264 (`-preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart`): one pass, nothing held in memory, odd sizes padded to even, ffmpeg errors surfaced, partial file deleted on failure. Deviation from the spec's mechanism ("H.264 via OpenCV, then re-mux"): OpenCV on Windows can't reliably *write* H.264 without a separately downloaded codec DLL, so frames go straight to ffmpeg; the result is the same (H.264 + faststart). Also `probe_codec` and an MP4 box reader (checks `moov` comes before `mdat`).
- `render/overlay.py`: skeleton, joint dots coloured by confidence (red < 0.3 ≤ amber < 0.7 ≤ green), live elbow and front-knee angle labels, a phase banner ("Ball 2: Arm circle"…), and a red tint on frames between arm-horizontal and release when legality is flagged. Handles several deliveries in one video.
- `render/keyframes.py`: PNGs at BFC, FFC, arm-horizontal, release and follow-through end, per delivery, with the overlay drawn.
- `render/clipping.py`: finds every release in a long clip (bowling-wrist speed peaks above the shoulder, ≥ 50% of the fastest, ≥ 3 s apart, refined to the wrist-height peak), cuts −2.5 s / +1.5 s windows, renumbers sub-sequences, and can write each clip as its own H.264 file (frame-accurate).
- `render/report_pdf.py`: page 1 has a per-delivery summary (technique score, injury band, legality with confidence, action type) and **both disclaimers** in a highlighted box, plus quality warnings. Each delivery gets a fault table with plain-language explanations, a keyframe strip, matplotlib charts (elbow with the 15° band, front knee, trunk lean; event markers) and the injury breakdown with notes. A coach section is included only when coach text is given (Phase 11). Every page footer repeats the disclaimers.
- `pipeline.analyze_session()`: splits into deliveries, analyses each, and runs session fatigue. `DeliveryAnalysis` moved into `schemas.py`.
- **CLI** now writes `result.json` (all deliveries; phases are numbered within each delivery's clip, and the clip's `start` maps them back to video frames), `pose.json`, `overlay.mp4`, `keyframes/delivery_NN/*.png`, `report.pdf`, `charts/`.

**Gates**
- **Overlay plays in Chrome**: verified in the built-in (Chromium) browser. The test page served from localhost fired loadedmetadata → canplay → playing with no error; duration 15.55 s (1,866 frames at 120 fps, exact); H.264 rated "probably" playable. Tests also check codec = h264 and moov-before-mdat.
- **The PDF opens**: pypdf reads it; it contains both disclaimers, FLAGGED legality, quality warnings and the charts.
- **Clipping splits a synthetic 3-ball video into 3**: releases within 1 frame. The CLI on that video: side view and right arm detected, every phase at confidence 1.0, ball 2 flagged (23.4°), ball 3 KNEE_COLLAPSE, 15 keyframes, a 7-page PDF.

**Tested**: 15 new tests (585 passed, 1 skipped; render coverage 90–100%).

**Bugs found and fixed**
- **Camera view on multi-ball videos**: the view was estimated over a window ending before the *fastest* delivery, which swallowed earlier follow-throughs (rotated shoulders look wide from the side) and returned "front" for a side-view clip. It now uses the *first* delivery's run-up.
- **Test data**: my stitched 3-ball video started the next run-up 1.0 s after a release (with a teleport), which polluted each clip's +1.5 s tail. Real bowlers walk back for many seconds; the test video now has a 2 s still gap.

**To remember for Phase 12**: Python's built-in test server doesn't support HTTP Range requests, so seeking failed in the browser check (the file itself is fine). The spec's `/files/{path}` endpoint must support Range so the player can seek.

**Note**: the full test suite now takes ~5 minutes (H.264 encoding + synthetic clips).

## Phase 7: Real-video front end ✅ (2026-11-20)

**Installed** (pinned in `ml/requirements.txt` from `pip freeze`): rtmlib 0.0.16, onnxruntime 1.32.0, mediapipe 1.2.3, ultralytics 8.4.215, **CPU** PyTorch 2.15.0 / torchvision 0.29.2 (from the PyTorch CPU index, a fraction of the CUDA build's size; GPU users install the matching CUDA build), and lap 0.5.14 (ByteTrack).

**Model weights**: `scripts/download_models.py` fetched 3 files (~67 MB) into `models/` (gitignored), with SHA-256 in `models/MANIFEST.json`: RTMPose-m Halpe-26 ONNX (from download.openmmlab.com, the URL rtmlib itself uses), `yolo11n.pt` (ultralytics/assets v8.3.0), `pose_landmarker_full.task` (MediaPipe models bucket). Halpe-26 was chosen because it includes **heels and big toes**, which the contact and no-ball checks need. **All three models load and run with `BOWLSENSE_OFFLINE=1`.**

**Done**
- `io/video.py`: probe (fps/size/frames), a frame iterator (the clip is never loaded fully; the capture is always released), single-frame read, and an image loader (handles non-ASCII Windows paths). Extension checks raise `UnsupportedMediaError`.
- `io/quality.py`: spec checks. < 60 fps → `LowFpsError` unless `--allow-low-fps`; 60–119 fps, < 720p, bowler in < 70% of frames, or mean keypoint confidence < 0.5 → warnings. **Camera-view estimate**: shoulder width ÷ body height in the run-up (narrow = side). Front vs behind comes from whether the bowler grows (approaching) or shrinks, with left/right shoulder order as the fallback. 6/6 correct on synthetic views × arms.
- `detect/person.py`: YOLO + ByteTrack person tracks (a fresh tracker per video; GPU if available, else CPU with a log line; telemetry sync off). Bowler = largest horizontal displacement + robust peak speed (95th percentile, so one ID-switch jump can't win); `--bowler-track-id` override. Missing weights → `ModelNotFoundError` telling you to run the download script.
- `pose/`: `PoseBackend` protocol, 20% crop padding, `load_backend("rtm" | "mediapipe")`. RTMPose maps Halpe-26 → our 21 names (big toe = toe); MediaPipe uses the Tasks API (1.x removed `mp.solutions`) and maps 33 landmarks → our 21 names, crop-normalised → full-frame. Scores are clamped to [0, 1].
- `pipeline.py`: `extract_pose()` (probe + quality → track → select bowler → pose per frame → detect arm → estimate view, or `--pose-from-json`, which checks size and fps match the video) and `analyze_delivery()` (phases + modules A–D + fatigue inputs).
- `cli.py`: `python -m bowlsense.cli analyze VIDEO --out DIR [--allow-low-fps] [--backend rtm|mediapipe] [--bowler-track-id N] [--pose-from-json FILE] [--config FILE] [-v]`. It writes `pose.json` + `result.json`. Errors print a clear message and exit with code 2. `--calib` and `--no-llm` arrive with the features they control (Phases 9 and 11).

**Tested**: 27 new tests (570 passed, 1 skipped).
- **Gate**: `cli analyze` on a rendered synthetic video with `--pose-from-json` detects the arm and side view by itself, phases within ±2, 22° flagged. `--backend mediapipe` (and `rtm`) through the real-video path: the real model loads and runs on every bowler frame, with ground-truth keypoints fed to the analysis (pose models don't detect stick figures).
- Plus: video I/O and errors, quality checks, view estimation, bowler selection (runner beats umpire/fielder, override, errors), keypoint mappings, crop padding, missing-model message, and **both backends + the YOLO tracker run with the network blocked**.
- `tests/data/real/` is empty, so the real-clip test skips. Drop clips there and it runs automatically.

**Bug found and fixed: the synthetic figure was mirror-imaged.** I'd declared "+Z = the bowler's left when facing +X", but in a right-handed frame the left is −Z, so every synthetic front/behind view was a mirror of what a real camera sees. Alignment's sign convention had been derived from those images, so on **real** front-view footage a right-armer's counter-rotation would have come out with the wrong sign. Fixed in the generator (left = −Z, chest = line × up, feet mirrored, truth sign) and in alignment (re-derived: front-on, a right-armer's bowling shoulder is on the image *left*). Side-view measurements weren't affected; all earlier tests pass.

**Also found**: ultralytics `pip install`s `lap` *at runtime* the first time ByteTrack runs. The offline guard blocked it (as designed); `lap` is now installed and pinned at setup.

**Not yet verified (needs real footage)**: pose accuracy, bowler selection and view estimation on real clips. The YOLO detection-unpacking lines are only exercised by real footage (a stick figure isn't detected as a person). The spec's evaluation script (Phase 16) is the place to measure this once clips are available.

## Phase 6: Modules D (injury risk) and E (fatigue + workload) ✅ (2026-11-20)

**Done**
- **D. Injury risk** (`analysis/injury.py`): a transparent rule-based model, exactly the spec's table. Each driver scores 0→100 linearly between its `injury.drivers` bounds (mixed action: yes = 100). A component = the mean of the drivers it could measure; overall = weighted sum, **renormalised over the components that had data**. Missing ones are listed in `notes` with the share of the model used (e.g. "uses 45% of the model" from a side-only clip), never counted as zero risk. Bands: < 35 low, 35–65 medium, > 65 high. Top 3 elevated drivers in plain words (lateral flexion feeds two components but is listed once). The disclaimer ("not a trained medical model … not medical advice") is always present. `injury_features()` builds the inputs from modules A–C plus three new measurements: arm slot (bowling arm angle from vertical at release, front view), peak upper-arm angular velocity (side view), and peak trunk rotation speed (front view).
- **E. Fatigue** (`analysis/fatigue.py`): the baseline is the mean of deliveries 1–6. Each later delivery gets the % change in run-up speed, release height, peak wrist speed and technique score; **only drops count** (a drop of `full_drop_pct` scores 100). The delivery index is the weighted mean over the metrics that were measured. The session index is the mean of the last 3. An alert fires on index > 40 or faults up by ≥ 2 vs baseline. With ≤ 6 deliveries it returns a note, not a trend. `delivery_fatigue_input()` extracts the per-delivery numbers (speeds in body heights per second, so camera distance doesn't matter).
- **E. Workload**: acute = balls in the last 7 days; chronic = balls in the last 28 days as a weekly average; ACWR flagged > 1.5. It needs ≥ 21 days of history, otherwise it returns `None` with a note explaining why. `Repository.bowler_ball_history()` gives (day, balls) per finished session from stored deliveries.
- Config: driver bounds, fatigue weights/full-drop points/windows, workload windows, all validated. Schemas: `InjuryFeatures`, `FatigueInput`, notes fields, and a per-delivery fatigue index (JSON Schema regenerated).

**Tested**: 55 new tests (543 total; injury 98%, fatigue 98% coverage).
- **Gate, monotonicity**: every driver is swept across twice its range in 40 random contexts (some drivers missing, as in single-view clips); its component(s) and the overall score never decrease. Mixed-action off→on never lowers lumbar (100 contexts).
- **Gate, fatigue drift**: each metric's drift is measured exactly (half the full drop → that metric's weighted 50). A progressive multi-metric drift gives a strictly rising index and an alert. Rises don't count; missing metrics drop out of the weights; there's a fault-increase alert; too few deliveries are handled; input order doesn't matter. A synthetic tiring session (release height falling, knee collapse rising) gives a rising index and an alert.
- Hand-computed full-model score; bounds/bands/top-driver tests; config validation; synthetic end-to-end (bad alignment raises lumbar by > 40; knee collapse + overstride raises knee by > 50; per-view feature availability). Workload hand-computed, steady vs spike, history requirement, future/old sessions; ball-history query (excludes failed sessions, other bowlers and the current session).

**Choices to know about**
- The spec's "arm slot extremeness" and "peak arm angular velocity **percentile**" have no population data to take a percentile from. They use configurable bounds instead (arm slot 20→60° from vertical; arm speed 1500→3000 °/s; trunk rotation 400→1000 °/s), documented in config as BowlSense defaults to tune in evaluation.
- The synthetic generator has no run-up-speed or arm-speed fatigue parameters, so those two drift metrics are verified with hand-built inputs rather than synthetic video.

## Phase 5: Modules A (legality), B (alignment), C (faults) ✅ (2026-11-20)

**Done**
- **A. Legality** (`analysis/legality.py`): elbow extension uses the spec formula (max over [AH, release] − min up to that max). Angles come from the **unsmoothed** track and are fitted with a cubic over the window (±0.02 s). The elbow straightens in ~50 ms (≈6 frames at 120 fps): general keypoint smoothing flattened it by ~20% (22° read as 17°), and raw max−min picked up noise. Confidence follows the spec's high/medium/low ladder, and is capped at low for non-side views. The note always carries the ICC 3D-testing disclaimer. Front-foot no-ball: given two popping-crease points, the front heel (toe if the heel is missing) must be on the bowler's side (taken from the back foot at BFC); on the line counts as a no-ball.
- **B. Alignment** (`analysis/alignment.py`): from a front or behind camera, rotation about the vertical axis narrows a body line rather than tilting it, so yaw = arcsin(signed width / full width). The full width is calibrated in the run-up (chest to camera), and each width is divided by that frame's trunk length so perspective (the bowler growing as they approach) cancels. Left/right image order gives the sign (past side-on = negative), with the correct sign for both arms and both frontal views. The action type at BFC is side-on / semi-open / front-on / mixed (closed ≤ 25°, open ≥ 55°, mixed if hip−shoulder > 30°). Counter-rotation = shoulder yaw at BFC − its minimum before FFC. The minimum is read on the unsmoothed track with a quadratic fit (the dip lasts < 0.1 s and smoothing filled it in). Hip–shoulder separation at FFC is also reported. Side view → all angles `None`, "needs front view".
- **C. Faults** (`analysis/faults.py`): all ten faults are table-driven (view, direction, threshold, measurement, plain-language explanation quoting the numbers). Severity ramps 0→100 from the threshold to 2× the threshold, or to an explicit `faults.severity_full` point for "too low" faults (UNDERSTRIDE 50%, LOW_RELEASE 100%, SHORT_FOLLOW 20%) and the zero-threshold WEAK_FRONT_ARM (10%). These are validated in config. `technique_score = 100 − weighted mean severity` over *measured* faults. Faults needing the other view go into `not_measured` with the reason, and are never scored as 0. New schemas: `AlignmentResult` and `FaultReport` (JSON Schema regenerated).

**Tested**: 132 new tests (488 total; coverage: alignment 98%, legality 98%, faults 95%).
- Section 10 gates, all for both arms: legality 8° → not flagged, 22° → flagged, measured within ±3° (120 and 240 fps). Counter-rotation within ±5° (4 action types × 3 amounts × 2 views). Mixed action classified as mixed (with 2 px noise). Every fault fires past its threshold and stays quiet below it.
- Plus: flag decisions under 2 px noise, confidence ladder, no-ball geometry (both arms, toe fallback, validation), hand-computed formula/classifier/severity tests, measurements vs generator truth, view routing, weighted score, missing-keypoint handling, and config validation.

**Accuracy and known limits** (measured on synthetic data, 540 px bowler)
- Elbow extension: clean data 100% within ±3°. With 1 px noise ≥ 92%; with 2 px noise 76–84% within ±3° (≈2.6° of angle noise per frame from 3 joints, and only 6–12 frames in the window, which is the information limit). The 8°/22° flag decision was correct in every run at every noise level.
- Counter-rotation: clean data 100% within ±5° (worst −3.4°); with 2 px noise 98%. There's a small consistent underestimate (mean −1.4°) from forward lean shortening the trunk during the run-up calibration. Action type was correct in all 240 runs.
- Knee collapse: clean data ±3°; with 2 px noise 88% within ±3°.
- Hip–shoulder separation at FFC is imprecise when the hips are near front-on (arcsin flattens there). It's reported, but no gate uses it.

**Deliberate choices / deviations**
- CHAIN_ORDER stays side-view as the spec says. Pelvis and trunk rotation timing come from the rate of change of apparent hip/shoulder width (most sensitive near front-on); the arm from peak wrist speed. On synthetic data this got the in-order verdict right in 15 of 16 cases, versus 14 of 16 with front-view yaw rates. It fires when the peaks are out of order by more than 25 ms.
- RUNUP_DECEL uses its own 0.1 s hip-speed derivative (phase detection's 0.25 s one leaked the post-BFC slowdown in: +2 points of bias).
- SHORT_FOLLOW measures hip travel from release to the detected follow-through end.

**Generator fixes found during this phase** (all earlier tests still pass)
- **The kinetic chain was wrong by default**: pelvis rotation peaked with or after the trunk, and well after release for front-on/mixed actions. Fixed: the pelvis turns fastest around FFC, then the trunk, then the arm. A new `pelvis_rotation_delay_s` breaks the order on purpose.
- **Front knee**: it was locked at 179.8° then jumped 8° at contact; it now lands at 178°. The pelvis now follows the scripted knee angle for the *whole* front-foot plant. Before, it switched curves at release and the near-straight knee jumped 11° in one frame (~1,400°/s).
- New truth fields: shoulder/hip yaw at BFC, separation at FFC, head offset, frontal trunk tilt, front-wrist height at FFC, and chain peak times.

**Note**: the full test suite now takes ~3 minutes (mostly synthetic clip generation). It may be worth caching generated clips across tests later.

## Phase 4: Heuristic phase detection ✅ (2026-11-20)

**Done** (`ml/bowlsense/phases/heuristic.py`, all thresholds in `config/default.yaml → phases`)
- **release**: peak bowling-wrist speed while the wrist is above the shoulder, refined to the wrist-height peak within ±3 frames.
- **arm_horizontal**: the last frame before release where the upper arm crosses horizontal going up.
- **ffc / bfc**: the ankle comes down and stays still near its lowest point for ≥3 frames, searching backward from release / ffc in the spec's windows. The impact itself is then pinned on the **unsmoothed** ankle track with a **change-point fit**: a least-squares fit of "descending (quadratic) curve, then flat ground", searched in 0.1-frame steps; contact = the first frame at or after impact.
- **run_up_start / follow_through_end**: hip horizontal speed crosses 20% / 15% of its peak and **stays there for 0.1 s**. Hip speed uses its own 0.25 s Savitzky–Golay derivative (hip translation is slow; this doesn't blur any joint).
- **bound**: the highest hip point between run-up start and bfc.
- **Ordering** is validated: out of order, or a fallback used → confidence 0.2, a logged warning, and the best guess is still returned. Front/behind views (the run-up is along the camera axis, so hip speed carries no information): run-up start / follow-through end = clip edges, confidence 0.5.
- `clean_sequence(..., smooth=False)` gives the gated + gap-filled but unsmoothed track that contact pinning needs. `savgol_derivative` is a shared low-noise derivative helper.
- `scripts/phase_error_matrix.py`: an evaluation tool that prints worst frame errors per event over 8 action variants × 10 seeds × 4 noise conditions × 2 bowler sizes.

**Results** (signed frame error vs ground truth, 120 fps, 80 runs per row):

| bowler height | condition | worst error, all 7 events | runs > ±2 frames |
|---|---|---|---|
| 540 px | clean / 2 px noise / 10% dropout / both | ±1 / ±2 / ±1 / ±2 | **0 of 320** |
| 300 px | clean / 10% dropout | ±1 | 0 |
| 300 px | 2 px noise (± dropout) | ffc +3, all others ±2 | ffc in ~9% of runs |

**Tested**: 66 new tests (356 total). **Section 10 gate**: ±2 frames at 120 fps with noise_px=2, both arms, 8 variants × 3 seeds; ±3 with 10% dropout; within ±1 on clean data; 240 fps is the same in time; front view (contacts/AH/release within ±2, confidence 0.5); fallback → low confidence + log; error cases; unit tests of the change-point fit (exact kink recovery, with NaN and noise), the sustained-crossing helper and release/arm-horizontal on an analytic arm swing.

**Known limit (documented, tested as a stress case)**: at 300 px bowler height, 2 px of noise is 0.67% of height while the foot moves only ~4 px/frame before impact. In ~9% of runs, noise makes two planted frames look like the foot is still descending, and front-foot contact slips to +3. That's at the information limit of the data, not an estimator bug. The gate is tested at **540 px**, a bowler filling about half of a 1080p frame as the recording tips ask (full body, side-on, ~10 m), where it passes 320/320. My generator's 300 px default was an arbitrary choice, not a spec number.

**Generator fixes found during this phase** (all Phase 2 tests still pass):
- **Feet now strike the ground moving down.** Swing height was eased to zero speed just before touchdown, like a foot being placed gently in mid-air, which made contacts look early.
- **Alignment is held steady for 40 ms into back-foot contact.** It was still turning at the contact instant, so "shoulder angle at BFC" depended on which frame landed on BFC.
- **The back foot can now always plant.** In overstride / high-release variants the pelvis was too far ahead, and the IK silently lifted the "planted" foot, which made the ground truth wrong. The pelvis is now lowered at BFC as needed, and the generator **refuses** any sequence where a planted foot leaves the ground within 4 frames of its contact. This also made a 105% stride reachable.

**Approaches tried and dropped** (so they aren't retried): single "still" onset (late by smoothing); a half-speed crossing (a fixed fraction fits one foot's speed curve but not the other's, so it was tuned to the generator); first frame near the ground (noise-limited); a linear line-fit (biased late because feet accelerate into impact).

## Phase 3: Geometry + smoothing + body ✅ (2026-11-19)

**Done**
- `geometry/angles.py`: joint angle (clipped `arccos`, never NaN on collinear points), segment angle / elevation / line tilt (image y-down handled; up = positive; range (−180, 180]), angular velocity (unwraps across ±180°), linear speed and velocity computed per valid run (NaN gaps are never bridged), `keypoint_array`/`keypoint_track` (missing = NaN), and `AngleSeries` conversion (NaN → None). fps ≤ 0 or NaN raises.
- `pose/smoothing.py`: confidence gating (`pose.min_conf`) → linear fill of interior gaps ≤ `max_gap_frames` (edges are never extrapolated; a filled sample gets the lower neighbour's confidence) → Savitzky–Golay per contiguous run (window = fps × `savgol_window_s`, rounded to odd; polyorder 2). It never smooths across a long gap. z is smoothed when present.
- `geometry/body.py`: role resolution (`joint(arm, "front", "ankle")` → `l_ankle` for a right-armer), bowling-arm detection (higher peak wrist speed), height estimate, `% of height` conversion, and `mirror_sequence` (x → width − x, l/r swapped, arm flipped; applying it twice is the identity).
- Config additions: `pose.savgol_window_s`, `pose.savgol_polyorder`, and a new `body` section.

**Tested**: 101 new tests (290 total). Hand-computed angles (90/180/0/45/120°, 3D, scale-invariance, float-noise collinearity), ±180° unwrap, NaN runs, gap filling at exactly/over the limit, no smoothing across gaps, polynomial preservation, noise reduction on synthetic data (noise and smoothing bias checked separately), dropout filling. **Gate: bowling-arm detection is 100% correct over 72 synthetic cases** (2 arms × 3 views × 4 noise/dropout levels × 3 seeds) with a margin over 1.3. Height is within 2% across views, arms, fps, noise and stride.

**Deliberate deviations from the spec (all in config, so they can be changed)**
- **Height factor 1.15, not 1.07, on the *vertical* ankle-to-nose distance.** With the spec's straight-line distance × 1.07, height came out 5% low in side view and 7% low in front view. The nose sits about 0.89 × stature above the ankle bone, so 1.07 can't be right, and every "% of height" metric (stride, release height, head offset) would have been inflated by 5–7%. The vertical distance is identical from every camera view. 1.15 = anthropometry plus typical running posture, rounded (not fitted exactly to the generator: −0.7% on synthetic data). Re-tune on real labelled clips in Phase 16.
- **Arm detection uses absolute wrist speed, as specified.** I tried wrist speed relative to the shoulder and relative to the pelvis: both were *worse* from the front and behind (trunk rotation moves the shoulders sideways), down to a wrong call at ratio 0.94.

**Generator fix found during this phase**: the synthetic front arm pulled down at 1,493°/s, nearly as fast as the bowling arm (2,188°/s). That's unrealistic: real front-arm pull-down is well under half. It's slowed to 768°/s. This raised the arm-detection margin from 1.1 to about 1.5 in front/behind views. All Phase 2 tests still pass.

## Phase 2: Synthetic generator + render_to_video ✅ (2026-11-19)

**Done** (`ml/synth/skeleton_gen.py`)
- A 3D stick figure with anthropometric segment lengths (Winter / Drillis-Contini ratios × height), driven through a keyframed action: standing → run-up (8 steps, legs cycle, arm swing) → bound → BFC → FFC → arm circle (~360°) → release → follow-through. Joint and body angles are interpolated with monotone cubic splines (PCHIP). Legs use 2-link IK towards scripted foot plants, so **segment lengths are exactly constant**.
- Orthographic projection to `side`, `front` or `behind` views. Left-arm bowlers are an exact mirror (Z→−Z, l/r names swapped).
- Controllable parameters: `fps, height_px, bowling_arm, view, action_type (side_on/semi_open/front_on/mixed), elbow_extension_deg, counter_rotation_deg, knee_flex_change_deg, stride_pct, lateral_flexion_deg, release_height_pct, runup_decel_pct, follow_through_pct, front_arm_ffc_deg, noise_px, dropout_rate, seed, include_z`.
- Outputs a `SynthResult`: the `PoseSequence`, ground-truth `PhaseEvents`, and a `SynthTruth` with every parameter **re-measured on the clean motion**. Ground truth uses the spec's own event definitions: contacts = the first frame with the foot down, release = wrist-height peak, arm-horizontal = the upper arm crossing horizontal going up, bound = the highest hip point, run-up start / follow-through end = hip-speed thresholds.
- **Fails loudly**: a parameter combination the body can't physically produce raises `ValueError` naming the parameter and the achievable range. It never silently returns a different value.
- `render_to_video(seq, path)` draws the stick figure to MP4 (mp4v; the browser-ready H.264 re-mux is Phase 8). Frames are drawn one at a time, and the writer is released in `finally`. `SKELETON_EDGES` was added to `schemas.py` for all renderers.
- ~0.15 s per 120-fps sequence.

**Tested**: 34 generator tests, 138 total, all green. Covers: constant segment lengths for side/front/behind × right/left; height scaling; exact left/right mirroring; the bowling wrist is fastest on the correct side; truth ≈ request across 5 parameter sets; elbow extension visible in the **2D** side view (the arm moves in the camera plane); release height from keypoints; mixed-action detection (hip–shoulder separation > 30° only for `mixed`); phase times independent of fps; feet on the ground at their contact frames; noise std and dropout rate; seed determinism; unreachable parameters raise; JSON round-trip; MP4 read back with the right frame count, size and fps.

Also fixed a real Phase 1 bug the full run caught: coach messages written within the same clock tick could come back out of order (the tie-break was a random UUID). They're now ordered by insertion, with a regression test that forces equal timestamps.

**Definitions later phases must match** (deliberate, documented choices)
- **Stride** = back ankle at BFC → front ankle at FFC (the standard delivery-stride definition). A literal same-frame ankle-to-ankle stride of 90%+ of height is anatomically out of reach (leg ≈ 49% of height). Phase 5's OVERSTRIDE/UNDERSTRIDE must use this definition.
- **Release height** = estimated ball height = wrist height + hand length (0.108 H). Wrist height alone can't reach the spec's 115% LOW_RELEASE threshold for a normal action. Phase 5 must use the same definition.
- Reachable ranges: stride 60–105%, release height ≈104–125% (depends on other params), run-up decel ≤ 50%, follow-through ≤ 120% of stride.
- Below 100 fps, only stride is checked against the request: the other quantities are measured at specific frames, and a coarse frame grid shifts them.

## Phase 1: Scaffold ✅ (2026-11-19, reworked for the 100%-local spec)

The spec was revised to be **100% local** (no Supabase, no hosted APIs, Ollama for the coach). Phase 1 was redone to match it. `CLAUDE.md` holds the current spec.

**Done**
- Layout: `config/`, `data/{storage/raw,storage/outputs,reference}/`, `models/`, `scripts/`, and the `ml/bowlsense/` subpackages (`io, detect, pose, geometry, phases, analysis, coach, render, db`) plus `ml/synth/`.
- Config: `config/default.yaml` holds the Section 14 defaults, plus video-quality thresholds and `jobs`/`api` sections, so no limits are hard-coded. `bowlsense/config.py` loads it into strict pydantic models. Unknown keys are rejected, and these are validated: fault/injury weight keys, the weight sum, band order, length-zone order and fps order. **The Ollama URL must be localhost, `server.host` must be loopback**, and storage paths resolve against the repo root. The only env vars are `BOWLSENSE_CONFIG` and `BOWLSENSE_OFFLINE`; there are no secrets.
- Schemas: `bowlsense/schemas.py` has every Section 4 contract, plus `FatigueDrift`, `FatigueResult` and `WorkloadResult` (named but not defined by the spec). JSON Schema export: `python -m bowlsense.schemas --out schema/bowlsense.schema.json`.
- Errors: `bowlsense/errors.py` defines exceptions with actionable messages.
- Database (`bowlsense/db/`): SQLAlchemy 2.x ORM (`models.py`) for the Section 8 tables. There's no auth: no `profiles`/`bowler_shares`/`owner_id`. Jobs also carry `progress` and `stage` for the SSE stream. `session.py` sets `foreign_keys=ON`, WAL, `busy_timeout`, and `BEGIN IMMEDIATE` for writes. The Alembic migration `0001` is applied automatically by `get_engine`; `alembic upgrade head` also works from `ml/`. `repo.py` handles bowlers, sessions, the atomic job claim (`UPDATE … RETURNING`), progress, retry→fail, coach messages, delete-session (rows + files) and delete-all-data.
- Storage: `bowlsense/storage.py` streams uploads in chunks to `raw/<uuid>.<ext>` (the original name is never used as a path) and enforces the size limit. It rejects path traversal, and handles session output folders, wipe and usage.
- Offline: `bowlsense/offline.py`. With `BOWLSENSE_OFFLINE=1`, importing `bowlsense` patches `socket` so any non-loopback connect/DNS lookup raises `OfflineViolationError`. Library telemetry env vars are set to off on import.
- `make check` → `scripts/check.py`: the **local-only URL check** (`scripts/check_urls.py`), ruff, black, mypy --strict, pytest+coverage, and the web checks once `web/` exists.

**Tested**: 103 tests, 98% coverage. Includes `alembic upgrade head` on an empty DB, migrations ≡ models (autogenerate diff is empty), downgrade, pragmas, the job queue with retries, 4 threads + **2 OS processes writing concurrently with no `database is locked`**, cascade delete of rows *and* files, delete-all, upload limits/traversal, the offline guard (blocks remote, allows loopback, installs on import), and the URL scanner.

**Known issues / notes**
- `make` isn't installed on this Windows machine; run `ml/.venv/Scripts/python scripts/check.py` instead.
- `ffmpeg` isn't installed yet; Phase 8 (overlay re-mux) needs it. Ollama isn't installed yet; Phase 11 needs it, and its tests will mock it.
- Python venv: `ml/.venv` (Python 3.11.9, created with `uv`). `scripts/setup.py` comes in Phase 16.

---

**Next**: Phase 15: image mode, learned phase model flag tests, optional 3D lift.
