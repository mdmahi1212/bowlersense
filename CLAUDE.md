# BowlSense — Build Prompt for Claude Code

> Paste this whole file into Claude Code as your first message, or save it as `CLAUDE.md` in an empty repo folder and say: **"Read CLAUDE.md and build the project, phase by phase."**

---

## 0. Your role and the goal

You are building **BowlSense**, an AI system that analyses cricket bowling from video and images. It's a university AI-course project. It must run end to end on a student laptop (CPU fallback) and faster on a GPU.

Input: a video (or a single image) of a bowler.
Output: an annotated skeleton video, per-delivery metrics, legality screening, technique faults, injury-risk score, ball speed and length, fatigue trends, comparison against a reference action, plain-language coaching, and a web dashboard that shows all of it.

**Build everything in this file. Do not skip modules. Do not leave TODOs, stubs, `pass`, or fake return values in finished code.**

---

## 1. Ground rules (read before writing any code)

1. **Work in phases (Section 12).** Finish one phase, run its checks, fix every failure, commit, then move on. Never start a phase while the previous one has failing tests.
2. **After every file you write, run it or its tests.** Don't batch up 10 files and test later.
3. **Pin every dependency** to the exact version you actually installed (`pip freeze` → `requirements.txt`, exact versions in `package.json`). Don't guess version numbers; install, verify the import works, then pin.
4. **Prefer libraries that install cleanly.** Specifically:
   - Use **`rtmlib`** (ONNX Runtime RTMPose/RTMDet) for pose. **Do NOT use `mmpose`/`mmcv`**: they break on install constantly.
   - Use **MediaPipe Pose** as a fallback backend behind the same interface.
   - Use **Ultralytics YOLO** for person and ball detection.
   - 3D lifting (MotionBERT) is optional and must sit behind a feature flag; the system must work fully without it.
5. **Everything configurable lives in `config/default.yaml`** (thresholds, weights, model names, fps requirements). No magic numbers buried in code.
6. **Type hints everywhere, docstrings on every public function.** Use `pydantic` models for every data structure that crosses a module boundary.
7. **Fail loudly and clearly.** Validate inputs at each stage. Raise custom exceptions (`LowFpsError`, `NoBowlerDetectedError`, `PoseQualityError`, etc.) with messages a user can act on. Never silently return zeros.
8. **Handle left-arm and right-arm bowlers.** Detect the bowling arm automatically (the arm with the highest peak wrist speed) and mirror logic accordingly. Write tests for both.
9. **Tests must not need the internet or real cricket videos.** Build a synthetic skeleton generator (Section 10) and test against it. Real-video tests are optional and skip if `tests/data/real/` is empty.
10. **If something is genuinely blocked** (a model won't download, a package won't install), stop, tell me exactly what failed and what you tried, and propose an alternative. Don't hack around it silently.
11. **Git:** init the repo, commit at the end of every phase with message `phase N: <summary>`.
12. **Keep a `PROGRESS.md`**, updated at the end of every phase: what's done, what's tested, known issues, next step.

---

## 2. Tech stack

| Layer | Choice |
|---|---|
| ML language | Python 3.11 |
| ML service | FastAPI + Uvicorn |
| Vision | OpenCV, Ultralytics YOLO, rtmlib (RTMPose, ONNX Runtime), MediaPipe (fallback) |
| Numerics | NumPy, SciPy, pandas |
| ML | scikit-learn, PyTorch (for phase model, bowler-ID model), tslearn (DTW) |
| Explainability | SHAP (only for the optional ML injury model) |
| Jobs | Simple DB-backed job queue (a `jobs` table polled by a worker) — no Redis required |
| PDF | WeasyPrint (if it fails to install on Windows, use ReportLab instead and note it) |
| LLM coach | Anthropic API (`claude-sonnet-5-5`), key from env `ANTHROPIC_API_KEY`; if missing, fall back to a template-based report |
| Frontend | Next.js (App Router, TypeScript), Tailwind CSS, Recharts, shadcn/ui |
| Backend data | Supabase (Postgres + Storage + Auth + Realtime) |
| Local dev DB | Supabase CLI local stack, OR plain SQLite fallback via env flag `BOWLSENSE_DB=sqlite` so the ML service can run without Supabase |
| Testing | pytest, pytest-cov; Vitest + Playwright for frontend smoke tests |
| Lint/format | ruff, black, mypy (Python); ESLint, Prettier (TS) |

---

## 3. Repository structure

Create exactly this layout:

```
bowlsense/
├── CLAUDE.md                  # this file
├── PROGRESS.md
├── README.md
├── .env.example
├── docker-compose.yml         # ml-service + worker (+ optional local supabase)
├── config/
│   └── default.yaml
├── ml/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── bowlsense/
│   │   ├── __init__.py
│   │   ├── config.py          # loads YAML into pydantic Settings
│   │   ├── schemas.py         # all pydantic models (Section 4)
│   │   ├── errors.py
│   │   ├── io/
│   │   │   ├── video.py       # read/write video, fps, frame iterator
│   │   │   └── quality.py     # recording quality checks
│   │   ├── detect/
│   │   │   ├── person.py      # YOLO + ByteTrack, bowler selection
│   │   │   └── ball.py        # ball detector + Kalman tracker
│   │   ├── pose/
│   │   │   ├── base.py        # PoseBackend protocol
│   │   │   ├── rtm.py         # rtmlib backend
│   │   │   ├── mediapipe_backend.py
│   │   │   ├── smoothing.py   # confidence gating, interpolation, Savitzky–Golay
│   │   │   └── lift3d.py      # optional MotionBERT, feature-flagged
│   │   ├── geometry/
│   │   │   ├── angles.py      # joint angles, segment angles, velocities
│   │   │   ├── body.py        # height estimate, bowling-arm detection, mirroring
│   │   │   └── pitch.py       # crease/pitch calibration, homography
│   │   ├── phases/
│   │   │   ├── heuristic.py   # rule-based phase detection (always available)
│   │   │   └── model.py       # learned phase model (BiLSTM), optional
│   │   ├── analysis/
│   │   │   ├── legality.py    # module A
│   │   │   ├── alignment.py   # module B
│   │   │   ├── faults.py      # module C
│   │   │   ├── injury.py      # module D
│   │   │   ├── fatigue.py     # module E
│   │   │   ├── delivery.py    # module F (ball speed, line, length, pitch map)
│   │   │   ├── compare.py     # module G (DTW, similarity, consistency)
│   │   │   └── bowler_id.py   # module G (sequence classifier)
│   │   ├── coach/
│   │   │   ├── drills.py      # fault → drill library
│   │   │   ├── llm.py         # Anthropic call + template fallback
│   │   │   └── prompts.py
│   │   ├── render/
│   │   │   ├── overlay.py     # skeleton + angle labels video
│   │   │   ├── keyframes.py   # BFC/FFC/release/follow-through stills
│   │   │   ├── clipping.py    # split long video into deliveries
│   │   │   └── report_pdf.py
│   │   ├── pipeline.py        # orchestrates everything → AnalysisResult
│   │   ├── db.py              # Supabase client OR SQLite fallback
│   │   ├── api.py             # FastAPI app
│   │   ├── worker.py          # job poller
│   │   └── cli.py             # `bowlsense analyze video.mp4 --out results/`
│   ├── synth/
│   │   └── skeleton_gen.py    # synthetic bowling skeleton generator (Section 10)
│   └── tests/
│       ├── conftest.py
│       ├── test_angles.py
│       ├── test_smoothing.py
│       ├── test_body.py
│       ├── test_phases.py
│       ├── test_legality.py
│       ├── test_alignment.py
│       ├── test_faults.py
│       ├── test_injury.py
│       ├── test_fatigue.py
│       ├── test_delivery.py
│       ├── test_compare.py
│       ├── test_pipeline_synthetic.py
│       ├── test_api.py
│       └── data/real/         # optional real clips; tests skip if empty
├── web/                       # Next.js app
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx                     # landing + upload
│   │   ├── sessions/page.tsx            # history list
│   │   ├── sessions/[id]/page.tsx       # session dashboard
│   │   ├── deliveries/[id]/page.tsx     # single-delivery deep dive
│   │   ├── compare/page.tsx             # user vs reference
│   │   ├── bowlers/page.tsx             # profiles
│   │   └── api/...                      # thin proxies to ML service where needed
│   ├── components/
│   │   ├── VideoPlayer.tsx              # frame-step, phase markers, overlay toggle
│   │   ├── AngleChart.tsx               # angle over time with threshold bands
│   │   ├── RiskGauge.tsx
│   │   ├── BodyRiskMap.tsx              # SVG skeleton coloured by risk
│   │   ├── PitchMap.tsx
│   │   ├── FaultList.tsx
│   │   ├── KeyframeStrip.tsx
│   │   ├── FatigueTrend.tsx
│   │   ├── CoachChat.tsx
│   │   └── UploadDropzone.tsx
│   └── lib/supabase.ts, lib/types.ts    # types generated to match schemas.py
└── supabase/
    └── migrations/0001_init.sql
```

---

## 4. Data contracts (`ml/bowlsense/schemas.py`)

Define these as pydantic v2 models. Every module consumes and produces these, and nothing else.

```python
Keypoint:         x: float, y: float, z: float | None, conf: float
FramePose:        frame_idx: int, t: float, keypoints: dict[str, Keypoint]   # names below
PoseSequence:     fps: float, width: int, height: int, view: Literal["side","front","behind","unknown"],
                  bowling_arm: Literal["right","left"], frames: list[FramePose]
PhaseEvents:      run_up_start, bound, bfc, ffc, arm_horizontal, release, follow_through_end: int  (frame indices)
                  confidence: float
BallTrack:        points: list[tuple[int, float, float, float]]   # frame, x, y, conf
                  release_frame: int | None, bounce_frame: int | None
AngleSeries:      name: str, unit: "deg", values: list[float | None], frames: list[int]
Fault:            code: str, name: str, severity: float (0–100), value: float, threshold: float,
                  frames: list[int], explanation: str
RiskComponent:    name: str, score: float (0–100), weight: float, drivers: list[str]
InjuryRisk:       overall: float, band: Literal["low","medium","high"], components: list[RiskComponent]
LegalityResult:   elbow_extension_deg: float, threshold_deg: float, flagged: bool,
                  confidence: Literal["low","medium","high"], note: str
DeliveryMetrics:  delivery_idx: int, phases: PhaseEvents, action_type: str, legality: LegalityResult,
                  faults: list[Fault], technique_score: float, injury: InjuryRisk,
                  speed_kmh: float | None, length: str | None, line: str | None,
                  pitch_xy: tuple[float,float] | None, release_height_pct: float,
                  stride_pct: float, angles: list[AngleSeries]
SessionResult:    session_id: str, deliveries: list[DeliveryMetrics], consistency_score: float,
                  fatigue: FatigueResult, workload: WorkloadResult, similar_pros: list[tuple[str,float]],
                  coach_report: str, quality_warnings: list[str], artifacts: dict[str, str]  # paths/URLs
```

Keypoint names (use these exact strings everywhere): `nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle, l_heel, r_heel, l_toe, r_toe`. Map each backend's output to these. If a backend lacks heel/toe, set them `None` and make every consumer handle that.

Write a script `web/scripts/gen-types.ts` (or a Python script) that exports the pydantic JSON Schema and generates matching TS types in `web/lib/types.ts`. Frontend types must never be hand-written duplicates.

---

## 5. Core pipeline (`pipeline.py`)

```
video → quality check → bowler detection + tracking → pose per frame (bowler crop)
      → keypoint cleaning → bowling-arm detection → (optional 3D lift)
      → delivery clipping (split multi-ball videos) → for each delivery:
           phases → angles → modules A–F → per-delivery metrics
      → session-level modules (consistency, fatigue, workload, comparison, bowler ID)
      → render overlay video, keyframes, charts data → coach report → PDF → SessionResult JSON
```

Details:

- **Quality check (`io/quality.py`)**: fps < 60 → hard error unless `--allow-low-fps`; 60–119 → warning "release metrics less reliable"; resolution < 720p → warning; bowler visible in < 70% of frames → warning; mean keypoint confidence < 0.5 → warning. Also estimate camera view (side vs front) from shoulder width / hip width ratio over the run-up.
- **Bowler selection**: among tracked persons, pick the track with the largest horizontal displacement + highest peak speed over the clip. Expose `--bowler-track-id` override.
- **Pose**: run on the bowler's bbox crop with 20% padding; map back to full-frame coordinates.
- **Smoothing (`pose/smoothing.py`)**: drop keypoints with conf < `pose.min_conf` (default 0.3); linear interpolate gaps ≤ `pose.max_gap_frames` (default 5); then Savitzky–Golay (window ≈ fps/10 rounded to odd, polyorder 2). Never smooth across gaps longer than max_gap; leave them `None`.
- **Body scale**: estimate standing height in pixels from ankle-to-nose distance during the run-up (median of top 20% frames) × 1.07. All distances are reported as % of height so they're camera-distance independent.
- **Bowling arm**: the side whose wrist hits the higher peak speed. Store in `PoseSequence.bowling_arm`. All "bowling_*" and "front_*" helpers resolve from this (right-arm bowler: front leg = left).

---

## 6. Phase detection (`phases/heuristic.py`, always on)

Using smoothed keypoints, with the bowling arm and front leg resolved:

- **release**: the frame of peak bowling-wrist *linear speed* while the wrist is above the shoulder; refine to the frame where wrist height peaks within ±3 frames.
- **arm_horizontal**: the last frame before release where the upper-arm (shoulder→elbow) angle to horizontal crosses 0° going upward (the arm is horizontal and rising).
- **ffc (front-foot contact)**: the front ankle's vertical velocity goes from downward to ~0 and stays ~0 for ≥ 3 frames, searching backward from release (window: release − 0.4 s to release).
- **bfc (back-foot contact)**: same test on the back ankle, searching backward from ffc (window: ffc − 0.35 s to ffc).
- **bound**: the minimum of hip height (highest jump point) between run-up and bfc.
- **run_up_start**: the first frame where hip horizontal speed > 20% of its peak.
- **follow_through_end**: the frame after release where hip speed drops below 15% of peak, or the end of the clip.
- **Validate ordering**: run_up_start < bound < bfc < ffc < arm_horizontal < release < follow_through_end. If violated, set `confidence` low, log which constraint failed, and still return the best guess.
- **Optional learned model (`phases/model.py`)**: a BiLSTM over normalised keypoint sequences that predicts a per-frame phase label. Train it from labelled data if present; the pipeline uses it only when `phases.use_model: true` and the weights file exists.

---

## 7. Analysis modules: exact specs

All thresholds come from `config/default.yaml`. Each module ships with a test using synthetic skeletons where the correct answer is known.

### A. Legality (`legality.py`)
- Elbow angle = angle at the elbow between the upper arm (shoulder→elbow) and the forearm (elbow→wrist), 180° = straight.
- `extension = max(angle over [arm_horizontal, release]) − min(angle over [arm_horizontal, the frame of that max])`. That is: how much the arm straightens from its most-flexed point after arm-horizontal up to release.
- `flagged = extension > 15.0` (configurable).
- confidence: high if fps ≥ 200 and mean elbow/wrist conf > 0.7; medium if fps ≥ 120; else low.
- `note` must always say: "2D video screening only. The official ICC test uses 3D motion capture."
- Front-foot no-ball (if crease calibrated): the front heel (or toe if heel is missing) behind the popping-crease line at ffc.

### B. Alignment (`alignment.py`)
- Shoulder-line angle and hip-line angle, in the image plane for a front view.
- At bfc: hip–shoulder difference → classify: side-on (shoulders and hips both closed), front-on (both open), semi-open (in between, matching), mixed (|shoulder − hip| > `mixed_threshold_deg`, default 30).
- Shoulder counter-rotation = the change in shoulder-line angle from bfc to the minimum before ffc. Flag if > 30°.
- Hip–shoulder separation at ffc.
- If the view is side-on, only report what's measurable from that view and mark the rest "needs front view".

### C. Faults (`faults.py`)
Each fault returns a `Fault` with severity 0–100 (a linear ramp from threshold to 2× threshold, clipped):

| code | rule (default threshold) | view |
|---|---|---|
| KNEE_COLLAPSE | front-knee flexion increase ffc→release > 20° | side |
| OVERSTRIDE | stride (ankle-to-ankle at ffc) > 90% of height | side |
| UNDERSTRIDE | stride < 65% of height | side |
| HEAD_FALL | head lateral offset from front foot at release > 15% of height | front |
| LATERAL_FLEXION | trunk side-bend at release > 30° | front |
| WEAK_FRONT_ARM | non-bowling wrist below shoulder height at ffc | side |
| RUNUP_DECEL | hip speed in final 3 strides drops > 15% from peak | side |
| LOW_RELEASE | release height < 115% of standing height | side |
| CHAIN_ORDER | peak angular velocity order not pelvis → trunk → arm | side |
| SHORT_FOLLOW | hip travel after release < 40% of stride length | side |

`technique_score = 100 − weighted mean severity` (weights in config).

### D. Injury risk (`injury.py`)
A transparent, rule-based weighted model. **It is not a trained medical model and must say so in the UI and the PDF.**

| component | drivers (score 0–100 each, linear between low/high bounds) | weight |
|---|---|---|
| lumbar | counter-rotation (10°→40°), mixed action (bool → 100), lateral flexion (15°→40°) | 0.40 |
| knee | knee collapse (0°→40°), overstride (85%→110% of height) | 0.20 |
| shoulder | arm slot extremeness, peak arm angular velocity percentile | 0.15 |
| side_strain | lateral flexion + trunk rotation speed | 0.15 |
| fatigue_modifier | session fatigue index (Module E) | 0.10 |

Component score = mean of its drivers; overall = weighted sum. Bands: < 35 low, 35–65 medium, > 65 high. Return the top 3 drivers in plain words. Optional: if `data/injury_labels.csv` exists, also train a logistic regression on the same features, report AUC, and show SHAP values. The rule model stays the default.

### E. Fatigue and workload (`fatigue.py`)
- Baseline = mean of deliveries 1–6. For each later delivery, compute the % change in run-up peak speed, release height, peak wrist speed, and technique score.
- Fatigue index (0–100) = a weighted combination of the negative drifts. Alert when > 40 or when the fault count increases by ≥ 2 versus baseline.
- Workload: balls per session, stored per bowler; acute (7-day) ÷ chronic (28-day rolling avg) ratio; flag > 1.5. Requires the DB history; returns `None` gracefully when there isn't enough history.

### F. Ball and delivery (`delivery.py`, `detect/ball.py`)
- Ball detection: YOLO `sports ball` class at low conf (0.1) inside a region of interest around the bowling hand at release, then forward in time. A Kalman filter (constant velocity + gravity in y) links detections and bridges ≤ 4 missed frames.
- Pitch calibration (`geometry/pitch.py`): the user clicks 4 points (the crease corners at both ends) once per session in the UI, OR auto-detect the white crease lines (Hough lines + the known 20.12 m × 3.05 m geometry). Compute a homography to a top-down pitch in metres.
- Speed = distance between the ball's projected positions over the first 5–10 frames after release ÷ time, × 3.6. Without calibration, return `None`, never a guess.
- Bounce frame = local max of the ball's y (image) / an abrupt change in vertical velocity.
- Length zones (metres from the batter's stumps): yorker < 2, full 2–6, good 6–8, short 8–10, bouncer > 10 (configurable). Line: relative to the stump line, with off/middle/leg/wide bands.
- Pitch map: an array of `pitch_xy` for all deliveries.
- Delivery type (stretch): a classifier on [speed, arm speed, wrist rotation proxy] → pace / spin; spin subtype only if hand keypoints are available.

### G. Comparison (`compare.py`, `bowler_id.py`)
- Feature vector per frame: normalised joint angles (elbow, knee, shoulder, trunk, hip–shoulder sep), time-normalised from bfc to follow-through.
- DTW (tslearn) between the user and each reference in `data/reference/*.json` (PoseSequence JSONs). Return per-joint deviation and an overall similarity % = `100 × exp(−dtw_dist / scale)`.
- Consistency score = 100 − normalised std of the key metrics across deliveries.
- Bowler ID: a small 1D-CNN or GRU over the feature sequences; train only if ≥ 3 bowlers with ≥ 20 deliveries each exist; otherwise the endpoint returns "insufficient data".
- Ship 2–3 reference sequences generated by the synthetic generator (labelled "Reference: textbook side-on" etc.). **Never ship anything named after a real player unless I add the data myself.**

### H. Coach (`coach/`)
- `drills.py`: a dict mapping each fault code → 2–3 drills (name, description, reps).
- `llm.py`: send ONLY the JSON metrics (no video, no images) to the Anthropic API with a system prompt: a friendly bowling coach, concrete, no medical diagnosis, cites the numbers, ≤ 250 words, prioritises the top 3 issues. Timeout 30 s, 2 retries; on any failure fall back to the template report built from faults + drills. Must work with no API key.
- The coach chat endpoint keeps the session metrics as context.

### I. Rendering (`render/`)
- Overlay video: skeleton lines, joint dots coloured by confidence, live elbow and front-knee angles, a phase label banner, a red tint on frames inside a flagged legality window. H.264 MP4 via OpenCV, then re-mux with ffmpeg (`-preset veryfast -movflags +faststart`) so browsers can play it.
- Keyframes: PNGs at bfc, ffc, arm_horizontal, release, follow-through.
- Clipping: detect each release in a long video, cut from release − 2.5 s to release + 1.5 s.
- PDF: summary page (scores, risk band, legality), fault table, keyframe strip, charts (matplotlib PNGs), coach report, disclaimers.

### J. Image mode
- A single image → pose → the subset of checks that work on one frame (posture at that phase: knee angle, trunk flexion, head position, arm slot). Ask the user which phase the image shows (dropdown: BFC / FFC / release), with "auto" as a best guess.

---

## 8. Database (`supabase/migrations/0001_init.sql`)

Tables (all with `id uuid pk default gen_random_uuid()`, `created_at timestamptz default now()`):

- `profiles` (user_id fk auth.users, display_name, role: 'player' | 'coach')
- `bowlers` (owner_id, name, height_cm, bowling_arm, bowling_type, dob nullable)
- `sessions` (bowler_id, recorded_at, video_path, fps, view, status: 'queued' | 'processing' | 'done' | 'failed', error text, calibration jsonb)
- `jobs` (session_id, status, attempts int, locked_at, last_error)
- `deliveries` (session_id, idx, phases jsonb, action_type, speed_kmh, length, line, pitch_x, pitch_y, technique_score, injury_overall, injury_band, legality_flagged, elbow_extension_deg)
- `delivery_metrics` (delivery_id, name, values jsonb) — the angle series
- `faults` (delivery_id, code, severity, value, threshold, frames int[])
- `session_summaries` (session_id, consistency, fatigue jsonb, workload jsonb, similar jsonb, coach_report text, pdf_path, overlay_path)
- `coach_messages` (session_id, role, content)

Enable RLS on all tables: users can only read/write rows belonging to their own bowlers. Coaches see bowlers shared with them via a `bowler_shares` table. Storage buckets: `raw-videos` (private), `outputs` (private, signed URLs).

The SQLite fallback in `db.py` mirrors the same tables (no RLS) for local/offline runs and tests.

---

## 9. API (`api.py`, FastAPI)

| method | path | does |
|---|---|---|
| GET | `/health` | model load status, GPU yes/no |
| POST | `/sessions` | create a session + job from an uploaded file or a storage path |
| GET | `/sessions/{id}` | status + SessionResult when done |
| POST | `/sessions/{id}/calibration` | save 4 pitch points |
| POST | `/analyze/image` | image mode, synchronous |
| GET | `/deliveries/{id}` | full DeliveryMetrics |
| POST | `/sessions/{id}/coach` | chat message → reply |
| GET | `/sessions/{id}/report.pdf` | PDF |
| GET | `/bowlers/{id}/workload` | ACWR history |

Max upload 500 MB; accept mp4/mov/avi/jpg/png; reject everything else with 415. CORS limited to the web origin from env. Write `test_api.py` with FastAPI's TestClient and the SQLite DB.

`worker.py`: polls `jobs` every 2 s, locks a row (`UPDATE … WHERE status='queued' … RETURNING`), runs the pipeline, writes results, retries up to 2 times, and stores the traceback in `last_error` on failure.

CLI: `python -m bowlsense.cli analyze path/to/video.mp4 --out out/ [--allow-low-fps] [--calib calib.json] [--no-llm]` must work with zero web/DB setup and write `result.json`, `overlay.mp4`, keyframes, and `report.pdf`.

---

## 10. Synthetic skeleton generator (`ml/synth/skeleton_gen.py`)

This is what makes testing possible without real footage. Build it carefully.

- A parametric 2D (and optional 3D) stick figure with segment lengths from standard anthropometric ratios × a height parameter.
- A keyframed bowling action: run-up (hip moves horizontally, legs cycle), bound, BFC, FFC, arm circle with the bowling arm rotating through ~360°, release, follow-through. Interpolate joint angles between keyframes with cubic splines.
- **Controllable parameters** (each module's test sets these and checks the module recovers them):
  `fps, height_px, bowling_arm, elbow_extension_deg, counter_rotation_deg, knee_flex_change_deg, stride_pct, lateral_flexion_deg, release_height_pct, runup_decel_pct, view ('side'|'front'), noise_px, dropout_rate, phase_frames (ground truth)`.
- Outputs a `PoseSequence` plus ground-truth `PhaseEvents` and parameter values.
- Also: `render_to_video(seq, path)` draws the stick figure to an MP4 so the full pipeline (including the video I/O and overlay code) can be smoke-tested. For that test, inject the keypoints directly via a `--pose-from-json` path so it doesn't depend on a pose model detecting a stick figure.

**Required tests using it:**
- Phases: ground-truth frames recovered within ±2 frames at 120 fps with noise_px=2.
- Legality: a set extension of 8° → not flagged; 22° → flagged; measured within ±3°.
- Alignment: counter-rotation recovered within ±5°; a mixed action is classified as mixed.
- Faults: each fault fires when its parameter is past the threshold and doesn't fire when it's below.
- Injury: monotonic — raising any driver never lowers its component score.
- Left-arm: the same tests pass with `bowling_arm='left'`.
- Robustness: 10% keypoint dropout still yields phases within ±3 frames.

---

## 11. Frontend (`web/`)

- **Upload page**: drag-and-drop, choose or create a bowler, recording tips (fps ≥ 120, tripod, side-on at ~10 m, full body in frame). Upload to Supabase Storage with a progress bar, then POST `/sessions`.
- **Session dashboard**: status via Supabase Realtime (queued → processing → done). When done: overall technique score, injury gauge + band, legality badge (with the screening disclaimer), consistency, fatigue trend chart, pitch map, a delivery list (click → detail), coach report, a PDF download.
- **Delivery detail**: a video player with the overlay toggle, frame stepping (← →), phase markers on the timeline, a keyframe strip, angle charts (elbow with the 15° band, front knee, trunk) synced to the player cursor, the fault list with "jump to frame", a body risk map, drills.
- **Calibration UI**: show the first frame and let the user click 4 crease corners; save to `/sessions/{id}/calibration`; re-run the delivery module.
- **Compare page**: pick a reference action; side-by-side synced skeleton playback; per-joint deviation bars.
- **Bowlers page**: profiles, session history, workload chart (ACWR line with the 0.8–1.3 safe band).
- **Coach chat**: a chat panel on the session page.
- **Design**: clean, sporty, dark-mode-first, accessible colour contrast; risk colours are green/amber/red, but always paired with text labels.
- Every page handles loading, empty and error states. No unhandled promise rejections.
- Auth: Supabase email magic link.

---

## 12. Phases and acceptance gates

Do these in order. **At the end of each phase:** run `ruff check`, `mypy`, `pytest` (and `npm run lint && npm run build && npm test` once `web/` exists), fix everything, update PROGRESS.md, commit.

| # | Phase | Done when |
|---|---|---|
| 1 | Scaffold: repo layout, config loader, schemas, errors, the SQLite DB, CI-style `make check` | `make check` passes on an empty test suite; schemas export JSON Schema |
| 2 | Synthetic generator + render_to_video | Generates side/front, left/right sequences; a test asserts segment lengths stay constant |
| 3 | Geometry + smoothing + body (angles, velocities, height, bowling arm, mirroring) | Unit tests with hand-computed angles pass; left/right arm detection 100% on synthetic data |
| 4 | Phase detection (heuristic) | Section 10 phase tests pass |
| 5 | Modules A, B, C | Section 10 tests pass |
| 6 | Module D + E | Monotonicity + fatigue drift tests pass |
| 7 | Real-video front end: video I/O, quality checks, YOLO person tracking, rtmlib pose (+ MediaPipe fallback), keypoint mapping | `cli analyze` runs on a real clip if one exists in tests/data/real; otherwise on the synthetic-video smoke test with `--pose-from-json`; a `--backend mediapipe` switch works |
| 8 | Rendering: overlay, keyframes, clipping, PDF | The overlay MP4 plays in Chrome (h264, faststart); the PDF opens; clipping splits a synthetic 3-ball video into 3 |
| 9 | Module F: ball tracking, calibration, speed, length/line, pitch map | Synthetic ball trajectory with known speed recovered within ±5%; returns None without calibration |
| 10 | Module G: DTW compare, consistency, bowler ID (with the insufficient-data path) | Identical sequences → ~100% similarity; tests pass |
| 11 | Coach: drills, LLM + template fallback, chat | Works with and without `ANTHROPIC_API_KEY` (mocked in tests) |
| 12 | Full pipeline + CLI + FastAPI + worker | `test_pipeline_synthetic.py` and `test_api.py` pass end to end |
| 13 | Supabase migration + RLS + storage | `supabase db reset` applies cleanly; an RLS test proves user A can't read user B's sessions |
| 14 | Next.js app: all pages in Section 11 | `npm run build` passes; a Playwright smoke test uploads a synthetic video and sees results |
| 15 | Image mode + optional learned phase model + optional 3D lift (feature-flagged) | Flags off: everything still passes; flags on: their own tests pass |
| 16 | Docker compose, README, final PROGRESS.md, evaluation script | `docker compose up` brings up the ML service + worker; `scripts/evaluate.py` prints the metrics table (pose PCK if labels exist, phase error, legality MAE, speed MAE, etc.) |

---

## 13. Error-prevention checklist (check this before every commit)

- [ ] No hardcoded paths; everything goes through config/env, using `pathlib`.
- [ ] Works on Windows and Linux (no shell-only commands in Python; use `pathlib` and `subprocess` with lists).
- [ ] Every keypoint access handles `None` (missing joint).
- [ ] Angles are computed with `np.arctan2` / a clipped `np.arccos`, never a raw `acos` on unclipped values (NaN guard).
- [ ] No division by zero (speeds over 0-length windows, zero height, zero fps).
- [ ] Frame indices are clamped to the sequence length.
- [ ] Left-arm mirroring tested.
- [ ] Video writer: frame size matches exactly; colour order is BGR; the writer is released in `finally`.
- [ ] Models load once (module-level cache), not per frame.
- [ ] Large videos are processed as a frame iterator, never loaded fully into RAM.
- [ ] The GPU is optional: detect CUDA and fall back to CPU with a log line.
- [ ] Model weights download on first run into `~/.cache/bowlsense` with a clear message; if offline, the error says which file to download manually and where to put it.
- [ ] Secrets only from env; `.env.example` lists every variable; `.env` is in `.gitignore`.
- [ ] The frontend never calls the ML service with the Supabase service-role key exposed to the browser.
- [ ] All disclaimers are present: legality = screening only; injury = not medical advice.
- [ ] `PROGRESS.md` updated.

---

## 14. Config defaults (`config/default.yaml`) — start from this

```yaml
video:
  min_fps_hard: 60
  min_fps_recommended: 120
pose:
  backend: rtm          # rtm | mediapipe
  min_conf: 0.3
  max_gap_frames: 5
  lift_3d: false
phases:
  use_model: false
legality:
  elbow_extension_threshold_deg: 15.0
alignment:
  mixed_threshold_deg: 30
  counter_rotation_flag_deg: 30
faults:
  knee_collapse_deg: 20
  overstride_pct: 90
  understride_pct: 65
  head_fall_pct: 15
  lateral_flexion_deg: 30
  low_release_pct: 115
  runup_decel_pct: 15
  short_follow_pct: 40
  weights: {KNEE_COLLAPSE: 1.5, OVERSTRIDE: 1, UNDERSTRIDE: 1, HEAD_FALL: 1, LATERAL_FLEXION: 1.5,
            WEAK_FRONT_ARM: 0.7, RUNUP_DECEL: 0.7, LOW_RELEASE: 0.7, CHAIN_ORDER: 1, SHORT_FOLLOW: 0.5}
injury:
  weights: {lumbar: 0.40, knee: 0.20, shoulder: 0.15, side_strain: 0.15, fatigue_modifier: 0.10}
  bands: {low: 35, high: 65}
fatigue:
  baseline_deliveries: 6
  alert_index: 40
workload:
  acwr_flag: 1.5
delivery:
  length_zones_m: {yorker: 2, full: 6, good: 8, short: 10}
coach:
  model: claude-sonnet-5-5
  max_words: 250
  timeout_s: 30
```

---

## 15. Final deliverables

1. A working CLI, API, worker and web app.
2. `pytest` coverage ≥ 80% for `ml/bowlsense/analysis`, `geometry` and `phases`.
3. README with: setup (Windows + Linux), how to record good footage, how to run the CLI, the web app and Docker, the architecture diagram (Mermaid), the module list, limitations and disclaimers.
4. `scripts/evaluate.py` producing the evaluation table for the course report.
5. `PROGRESS.md` showing every phase complete.

**Start now with Phase 1. Before writing code, print a short plan for Phase 1, then build it, test it and commit it. Keep going phase by phase without waiting for me unless you're blocked.**
