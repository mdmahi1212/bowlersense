"""Load ``config/default.yaml`` into validated pydantic settings, plus environment settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bowlsense.errors import ConfigError

FAULT_CODES: tuple[str, ...] = (
    "KNEE_COLLAPSE",
    "OVERSTRIDE",
    "UNDERSTRIDE",
    "HEAD_FALL",
    "LATERAL_FLEXION",
    "WEAK_FRONT_ARM",
    "RUNUP_DECEL",
    "LOW_RELEASE",
    "CHAIN_ORDER",
    "SHORT_FOLLOW",
)
INJURY_COMPONENTS: tuple[str, ...] = (
    "lumbar",
    "knee",
    "shoulder",
    "side_strain",
    "fatigue_modifier",
)
INJURY_DRIVERS: tuple[str, ...] = (
    "counter_rotation_deg",
    "lateral_flexion_deg",
    "knee_collapse_deg",
    "stride_pct",
    "arm_slot_deg",
    "arm_angular_velocity_deg_s",
    "trunk_rotation_speed_deg_s",
    "fatigue_index",
)
FATIGUE_METRICS: tuple[str, ...] = (
    "runup_speed",
    "release_height",
    "wrist_speed",
    "technique_score",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VideoSettings(_Section):
    """Recording-quality requirements."""

    min_fps_hard: float = Field(gt=0)
    min_fps_recommended: float = Field(gt=0)
    min_height_px: int = Field(gt=0)
    min_bowler_visible_frac: float = Field(ge=0, le=1)
    min_mean_keypoint_conf: float = Field(ge=0, le=1)
    front_view_width_ratio: float = Field(gt=0)
    approach_size_change: float = Field(gt=0)
    slowmo_expected_arm_dps: float = Field(gt=0)
    slowmo_expected_wrist_hps: float = Field(gt=0)
    slowmo_min_factor: float = Field(gt=1)
    slowmo_factors: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _fps_order(self) -> VideoSettings:
        if self.min_fps_recommended < self.min_fps_hard:
            raise ValueError("min_fps_recommended must be >= min_fps_hard")
        return self


class DetectSettings(_Section):
    """Person (and later ball) detection and tracking."""

    yolo_model: str = Field(min_length=1)
    person_conf: float = Field(gt=0, lt=1)
    tracker: str = Field(min_length=1)
    min_track_frames: int = Field(ge=1)
    selector_model: str = Field(min_length=1)
    stitch_max_gap_s: float = Field(ge=0)
    stitch_max_jump: float = Field(gt=0)
    stitch_max_size_ratio: float = Field(gt=1)
    colour_window_s: float = Field(gt=0)
    colour_split_threshold: float = Field(gt=0, le=1)
    colour_join_threshold: float = Field(gt=0, le=1)


class PoseSettings(_Section):
    """Pose backend and keypoint cleaning."""

    backend: Literal["rtm", "mediapipe"]
    crop_padding: float = Field(ge=0, le=1)
    rtm_model: str = Field(min_length=1)
    rtm_input_size: tuple[int, int]
    mediapipe_model: str = Field(min_length=1)
    min_conf: float = Field(ge=0, le=1)
    max_gap_frames: int = Field(ge=0)
    savgol_window_s: float = Field(gt=0)
    savgol_polyorder: int = Field(ge=1, le=5)
    lift_3d: bool
    lift_model: str = Field(min_length=1)
    lift_window: int = Field(ge=9)


class BodySettings(_Section):
    """Body-scale estimation."""

    height_factor: float = Field(gt=0)
    height_top_frac: float = Field(gt=0, le=1)
    runup_end_before_peak_s: float = Field(ge=0)
    ankle_height_pct: float = Field(ge=0)
    hand_length_pct: float = Field(ge=0)


class PhaseSettings(_Section):
    """Phase detection options."""

    use_model: bool
    model_agreement_s: float = Field(ge=0)
    model_file: str = Field(min_length=1)
    release_refine_frames: int = Field(ge=0)
    ffc_window_s: float = Field(gt=0)
    bfc_window_s: float = Field(gt=0)
    contact_speed_h_per_s: float = Field(gt=0)
    contact_min_frames: int = Field(ge=1)
    contact_ground_tol_pct: float = Field(gt=0)
    run_up_speed_frac: float = Field(gt=0, lt=1)
    follow_through_speed_frac: float = Field(gt=0, lt=1)
    speed_hold_s: float = Field(ge=0)
    hip_speed_smooth_s: float = Field(gt=0)
    contact_velocity_window_s: float = Field(gt=0)
    contact_fit_s: float = Field(gt=0)
    contact_fit_order: int = Field(ge=1, le=2)
    low_confidence: float = Field(ge=0, le=1)
    no_horizontal_motion_confidence: float = Field(ge=0, le=1)


class LegalitySettings(_Section):
    """Elbow-extension screening."""

    elbow_extension_threshold_deg: float = Field(gt=0)
    high_conf_fps: float = Field(gt=0)
    high_conf_keypoint: float = Field(ge=0, le=1)
    medium_conf_fps: float = Field(gt=0)
    fit_degree: int = Field(ge=1, le=5)
    fit_pad_s: float = Field(ge=0)
    screen_min_fps: float = Field(gt=0)
    screen_views: tuple[Literal["side", "front", "behind", "unknown"], ...] = Field(min_length=1)
    screen_min_phase_confidence: float = Field(ge=0, le=1)
    screen_max_window_s: float = Field(gt=0)
    fallback_window_s: float = Field(gt=0)
    screen_max_plausible_deg: float = Field(gt=0)


class AlignmentSettings(_Section):
    """Action-type classification thresholds."""

    mixed_threshold_deg: float = Field(gt=0)
    counter_rotation_flag_deg: float = Field(gt=0)
    closed_max_deg: float = Field(ge=0, le=90)
    open_min_deg: float = Field(ge=0, le=90)
    width_calib_top_frac: float = Field(gt=0, le=1)
    min_fit_half_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _order(self) -> AlignmentSettings:
        if self.closed_max_deg >= self.open_min_deg:
            raise ValueError("alignment.closed_max_deg must be < open_min_deg")
        return self


class FaultSettings(_Section):
    """Technique-fault thresholds and score weights."""

    knee_collapse_deg: float = Field(gt=0)
    overstride_pct: float = Field(gt=0)
    understride_pct: float = Field(gt=0)
    head_fall_pct: float = Field(gt=0)
    lateral_flexion_deg: float = Field(gt=0)
    low_release_pct: float = Field(gt=0)
    runup_decel_pct: float = Field(gt=0)
    short_follow_pct: float = Field(gt=0)
    weak_front_arm_pct: float = Field(ge=0)
    chain_order_tolerance_s: float = Field(ge=0)
    runup_final_window_s: float = Field(gt=0)
    runup_speed_smooth_s: float = Field(gt=0)
    knee_fit_degree: int = Field(ge=1, le=4)
    chain_window_after_release_s: float = Field(gt=0)
    severity_full: dict[str, float]
    weights: dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> FaultSettings:
        if set(self.weights) != set(FAULT_CODES):
            missing = set(FAULT_CODES) - set(self.weights)
            extra = set(self.weights) - set(FAULT_CODES)
            raise ValueError(f"faults.weights keys mismatch: missing={missing} extra={extra}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("faults.weights must be non-negative")
        if self.understride_pct >= self.overstride_pct:
            raise ValueError("understride_pct must be < overstride_pct")
        unknown = set(self.severity_full) - set(FAULT_CODES)
        if unknown:
            raise ValueError(f"faults.severity_full has unknown codes: {unknown}")
        needed = {"UNDERSTRIDE", "LOW_RELEASE", "SHORT_FOLLOW", "WEAK_FRONT_ARM"}
        missing = needed - set(self.severity_full)
        if missing:
            raise ValueError(f"faults.severity_full must set: {sorted(missing)}")
        if self.severity_full["UNDERSTRIDE"] >= self.understride_pct:
            raise ValueError("severity_full.UNDERSTRIDE must be below understride_pct")
        if self.severity_full["LOW_RELEASE"] >= self.low_release_pct:
            raise ValueError("severity_full.LOW_RELEASE must be below low_release_pct")
        if self.severity_full["SHORT_FOLLOW"] >= self.short_follow_pct:
            raise ValueError("severity_full.SHORT_FOLLOW must be below short_follow_pct")
        if self.severity_full["WEAK_FRONT_ARM"] <= self.weak_front_arm_pct:
            raise ValueError("severity_full.WEAK_FRONT_ARM must exceed weak_front_arm_pct")
        return self


class InjuryBands(_Section):
    """Overall-risk band cut-offs (0-100)."""

    low: float = Field(ge=0, le=100)
    high: float = Field(ge=0, le=100)


class InjurySettings(_Section):
    """Rule-based injury-risk model weights."""

    weights: dict[str, float]
    bands: InjuryBands
    drivers: dict[str, tuple[float, float]]

    @model_validator(mode="after")
    def _check(self) -> InjurySettings:
        if set(self.weights) != set(INJURY_COMPONENTS):
            raise ValueError(f"injury.weights must have exactly the keys {INJURY_COMPONENTS}")
        if set(self.drivers) != set(INJURY_DRIVERS):
            raise ValueError(f"injury.drivers must have exactly the keys {INJURY_DRIVERS}")
        bad = [k for k, (lo, hi) in self.drivers.items() if not lo < hi]
        if bad:
            raise ValueError(f"injury.drivers bounds must be [low, high] with low < high: {bad}")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("injury.weights must sum to 1.0")
        if self.bands.low >= self.bands.high:
            raise ValueError("injury.bands.low must be < injury.bands.high")
        return self


class FatigueSettings(_Section):
    """Fatigue drift detection."""

    baseline_deliveries: int = Field(ge=1)
    alert_index: float = Field(ge=0, le=100)
    recent_deliveries: int = Field(ge=1)
    fault_increase_alert: int = Field(ge=1)
    full_drop_pct: dict[str, float]
    weights: dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> FatigueSettings:
        for name, table in (("full_drop_pct", self.full_drop_pct), ("weights", self.weights)):
            if set(table) != set(FATIGUE_METRICS):
                raise ValueError(f"fatigue.{name} must have exactly the keys {FATIGUE_METRICS}")
        if any(v <= 0 for v in self.full_drop_pct.values()):
            raise ValueError("fatigue.full_drop_pct values must be positive")
        if any(v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("fatigue.weights must be non-negative with a positive sum")
        return self


class WorkloadSettings(_Section):
    """Acute:chronic workload ratio."""

    acwr_flag: float = Field(gt=0)
    acute_days: int = Field(ge=1)
    chronic_days: int = Field(ge=1)
    min_history_days: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> WorkloadSettings:
        if self.acute_days >= self.chronic_days:
            raise ValueError("workload.acute_days must be < chronic_days")
        return self


class LengthZones(_Section):
    """Upper bounds (metres from the batter's stumps) of each length zone."""

    yorker: float = Field(gt=0)
    full: float = Field(gt=0)
    good: float = Field(gt=0)
    short: float = Field(gt=0)

    @model_validator(mode="after")
    def _increasing(self) -> LengthZones:
        if not self.yorker < self.full < self.good < self.short:
            raise ValueError("length zones must be strictly increasing: yorker<full<good<short")
        return self


class DeliverySettings(_Section):
    """Ball and delivery analysis."""

    length_zones_m: LengthZones
    type_model_file: str = Field(min_length=1)
    speed_frames: tuple[int, int]
    first_ball_within_frames: int = Field(ge=0)
    line_bands_m: dict[str, float]
    batter_hand: Literal["right", "left"]
    position_error_px: float = Field(gt=0)
    max_length_error_m: float = Field(gt=0)
    max_line_error_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> DeliverySettings:
        lo, hi = self.speed_frames
        if not 2 <= lo <= hi:
            raise ValueError("delivery.speed_frames must be [min, max] with 2 <= min <= max")
        if set(self.line_bands_m) != {"middle", "stump", "wide"}:
            raise ValueError("delivery.line_bands_m needs exactly middle, stump, wide")
        b = self.line_bands_m
        if not 0 < b["middle"] < b["stump"] < b["wide"]:
            raise ValueError("delivery.line_bands_m must satisfy 0 < middle < stump < wide")
        return self


class CompareSettings(_Section):
    """Module G: comparison with reference actions and consistency."""

    resample_points: int = Field(ge=10)
    similarity_scale: float = Field(gt=0)
    reference_dir: Path
    consistency_metrics: tuple[str, ...] = Field(min_length=1)

    @property
    def reference_path(self) -> Path:
        """Absolute folder of reference action JSONs."""
        return _resolve(self.reference_dir)


class BowlerIdSettings(_Section):
    """Module G: bowler identification."""

    min_bowlers: int = Field(ge=2)
    min_deliveries_per_bowler: int = Field(ge=2)
    epochs: int = Field(ge=1)
    hidden: int = Field(ge=4)


class BallSettings(_Section):
    """Ball detection and tracking."""

    yolo_class: int = Field(ge=0)
    det_conf: float = Field(gt=0, lt=1)
    roi_px: int = Field(ge=64)
    max_missed: int = Field(ge=0)
    gate_px: float = Field(gt=0)
    meas_noise_px: float = Field(gt=0)
    accel_noise_px_s2: float = Field(gt=0)
    track_after_release_s: float = Field(gt=0)


class JobSettings(_Section):
    """Background job queue."""

    poll_interval_s: float = Field(gt=0)
    max_attempts: int = Field(ge=1)


class ApiSettings(_Section):
    """HTTP API limits."""

    max_upload_mb: int = Field(gt=0)
    allowed_extensions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _dotted_lower(self) -> ApiSettings:
        bad = [e for e in self.allowed_extensions if not e.startswith(".") or e != e.lower()]
        if bad:
            raise ValueError(f"allowed_extensions must be lowercase and start with '.': {bad}")
        return self


LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def is_local_url(url: str) -> bool:
    """Return True if ``url`` is an http(s) URL pointing at this machine."""
    parsed = urlsplit(url)
    return parsed.scheme in ("http", "https") and (parsed.hostname or "") in LOCAL_HOSTS


class RenderSettings(_Section):
    """Overlay video and keyframe rendering."""

    ffmpeg_preset: str = Field(min_length=1)
    crf: int = Field(ge=0, le=51)
    conf_low: float = Field(ge=0, le=1)
    conf_high: float = Field(ge=0, le=1)
    tint_alpha: float = Field(ge=0, le=1)
    keyframe_events: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> RenderSettings:
        if self.conf_low >= self.conf_high:
            raise ValueError("render.conf_low must be < conf_high")
        allowed = {
            "run_up_start",
            "bound",
            "bfc",
            "ffc",
            "arm_horizontal",
            "release",
            "follow_through_end",
        }
        bad = set(self.keyframe_events) - allowed
        if bad:
            raise ValueError(f"render.keyframe_events has unknown events: {sorted(bad)}")
        return self


class ClippingSettings(_Section):
    """Splitting a long video into one clip per delivery."""

    before_release_s: float = Field(gt=0)
    after_release_s: float = Field(gt=0)
    min_gap_s: float = Field(gt=0)
    min_peak_frac: float = Field(gt=0, le=1)


class CoachSettings(_Section):
    """Local LLM coach (Ollama). Never configured with a remote URL."""

    provider: Literal["ollama"]
    ollama_url: str
    model: str = Field(min_length=1)
    max_words: int = Field(gt=0)
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    top_issues: int = Field(ge=1)
    chat_history_turns: int = Field(ge=0)

    @model_validator(mode="after")
    def _local_only(self) -> CoachSettings:
        if not is_local_url(self.ollama_url):
            raise ValueError(
                f"coach.ollama_url must point at localhost/127.0.0.1, got '{self.ollama_url}'"
            )
        return self


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


class ImageSettings(_Section):
    """Image mode: single-photo posture checks."""

    front_knee_min_deg: float = Field(gt=90, le=180)
    arm_slot_high_deg: float = Field(ge=0, le=90)
    arm_slot_round_deg: float = Field(ge=0, le=180)
    front_hip_width_ratio: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _slots_ordered(self) -> ImageSettings:
        if self.arm_slot_high_deg >= self.arm_slot_round_deg:
            raise ValueError("image.arm_slot_high_deg must be below arm_slot_round_deg")
        return self


class StorageSettings(_Section):
    """Local data locations. Relative paths resolve against the repo root."""

    root: Path
    db_path: Path
    models_dir: Path

    @property
    def root_path(self) -> Path:
        """Absolute storage root (holds ``raw/`` and ``outputs/``)."""
        return _resolve(self.root)

    @property
    def db_file(self) -> Path:
        """Absolute SQLite database path."""
        return _resolve(self.db_path)

    @property
    def models_path(self) -> Path:
        """Absolute model-weights folder."""
        return _resolve(self.models_dir)


class ServerSettings(_Section):
    """Local servers. Bound to the loopback interface only."""

    host: str
    api_port: int = Field(ge=1, le=65535)
    web_port: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def _loopback(self) -> ServerSettings:
        if self.host not in LOCAL_HOSTS:
            raise ValueError(f"server.host must be a loopback address, got '{self.host}'")
        if self.api_port == self.web_port:
            raise ValueError("server.api_port and server.web_port must differ")
        return self

    @property
    def web_origins(self) -> list[str]:
        """Origins the API accepts for CORS."""
        return [
            f"http://localhost:{self.web_port}",
            f"http://127.0.0.1:{self.web_port}",
        ]


class Settings(_Section):
    """All YAML-driven settings."""

    video: VideoSettings
    detect: DetectSettings
    pose: PoseSettings
    body: BodySettings
    phases: PhaseSettings
    legality: LegalitySettings
    alignment: AlignmentSettings
    faults: FaultSettings
    injury: InjurySettings
    fatigue: FatigueSettings
    workload: WorkloadSettings
    delivery: DeliverySettings
    compare: CompareSettings
    bowler_id: BowlerIdSettings
    ball: BallSettings
    jobs: JobSettings
    api: ApiSettings
    render: RenderSettings
    clipping: ClippingSettings
    coach: CoachSettings
    image: ImageSettings
    storage: StorageSettings
    server: ServerSettings


class EnvSettings(_Section):
    """Settings read from environment variables. There are no secrets: everything is local."""

    config_path: Path
    offline: bool


def load_settings(path: Path | str | None = None) -> Settings:
    """Load and validate settings from a YAML file.

    Args:
        path: YAML file to load. Defaults to ``$BOWLSENSE_CONFIG`` or ``config/default.yaml``.

    Raises:
        ConfigError: if the file is missing, not valid YAML, or fails validation.
    """
    cfg_path = Path(path) if path is not None else load_env().config_path
    if not cfg_path.is_file():
        raise ConfigError(
            f"Config file not found: {cfg_path}. Set BOWLSENSE_CONFIG or restore "
            "config/default.yaml."
        )
    try:
        raw: Any = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {cfg_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {cfg_path} must contain a YAML mapping at the top level.")
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {cfg_path}:\n{exc}") from exc


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def offline_requested() -> bool:
    """Return True when ``BOWLSENSE_OFFLINE`` is set to a truthy value."""
    return (_env("BOWLSENSE_OFFLINE") or "").lower() in TRUE_VALUES


def load_env() -> EnvSettings:
    """Read environment-driven settings, applying defaults for unset variables."""
    return EnvSettings(
        config_path=Path(_env("BOWLSENSE_CONFIG") or DEFAULT_CONFIG_PATH),
        offline=offline_requested(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return load_settings()
