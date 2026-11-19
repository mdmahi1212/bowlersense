# BowlSense — Progress

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

**Next**: Phase 7: real-video front end (video I/O, quality checks, YOLO person tracking, rtmlib pose + MediaPipe fallback, keypoint mapping, `cli analyze` with `--pose-from-json`). First phase that downloads model weights (via `scripts/download_models.py`).
