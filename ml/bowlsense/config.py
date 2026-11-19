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

    @model_validator(mode="after")
    def _fps_order(self) -> VideoSettings:
        if self.min_fps_recommended < self.min_fps_hard:
            raise ValueError("min_fps_recommended must be >= min_fps_hard")
        return self


class PoseSettings(_Section):
    """Pose backend and keypoint cleaning."""

    backend: Literal["rtm", "mediapipe"]
    min_conf: float = Field(ge=0, le=1)
    max_gap_frames: int = Field(ge=0)
    savgol_window_s: float = Field(gt=0)
    savgol_polyorder: int = Field(ge=1, le=5)
    lift_3d: bool


class BodySettings(_Section):
    """Body-scale estimation."""

    height_factor: float = Field(gt=0)
    height_top_frac: float = Field(gt=0, le=1)
    runup_end_before_peak_s: float = Field(ge=0)


class PhaseSettings(_Section):
    """Phase detection options."""

    use_model: bool
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


class AlignmentSettings(_Section):
    """Action-type classification thresholds."""

    mixed_threshold_deg: float = Field(gt=0)
    counter_rotation_flag_deg: float = Field(gt=0)


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
        return self


class InjuryBands(_Section):
    """Overall-risk band cut-offs (0-100)."""

    low: float = Field(ge=0, le=100)
    high: float = Field(ge=0, le=100)


class InjurySettings(_Section):
    """Rule-based injury-risk model weights."""

    weights: dict[str, float]
    bands: InjuryBands

    @model_validator(mode="after")
    def _check(self) -> InjurySettings:
        if set(self.weights) != set(INJURY_COMPONENTS):
            raise ValueError(f"injury.weights must have exactly the keys {INJURY_COMPONENTS}")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("injury.weights must sum to 1.0")
        if self.bands.low >= self.bands.high:
            raise ValueError("injury.bands.low must be < injury.bands.high")
        return self


class FatigueSettings(_Section):
    """Fatigue drift detection."""

    baseline_deliveries: int = Field(ge=1)
    alert_index: float = Field(ge=0, le=100)


class WorkloadSettings(_Section):
    """Acute:chronic workload ratio."""

    acwr_flag: float = Field(gt=0)


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


class CoachSettings(_Section):
    """Local LLM coach (Ollama). Never configured with a remote URL."""

    provider: Literal["ollama"]
    ollama_url: str
    model: str = Field(min_length=1)
    max_words: int = Field(gt=0)
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)

    @model_validator(mode="after")
    def _local_only(self) -> CoachSettings:
        if not is_local_url(self.ollama_url):
            raise ValueError(
                f"coach.ollama_url must point at localhost/127.0.0.1, got '{self.ollama_url}'"
            )
        return self


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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
    jobs: JobSettings
    api: ApiSettings
    coach: CoachSettings
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
