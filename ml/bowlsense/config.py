"""Load ``config/default.yaml`` into validated pydantic settings, plus environment settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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
    lift_3d: bool


class PhaseSettings(_Section):
    """Phase detection options."""

    use_model: bool


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


class CoachSettings(_Section):
    """LLM coach."""

    model: str = Field(min_length=1)
    max_words: int = Field(gt=0)
    timeout_s: float = Field(gt=0)


class Settings(_Section):
    """All YAML-driven settings."""

    video: VideoSettings
    pose: PoseSettings
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


class EnvSettings(_Section):
    """Settings read from environment variables (secrets and deployment paths)."""

    config_path: Path
    db_backend: Literal["sqlite", "supabase"]
    sqlite_path: Path
    cache_dir: Path
    anthropic_api_key: str | None
    supabase_url: str | None
    supabase_service_role_key: str | None
    web_origin: str


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


def load_env() -> EnvSettings:
    """Read environment-driven settings, applying defaults for unset variables.

    Raises:
        ConfigError: if ``BOWLSENSE_DB`` has an unknown value.
    """
    cache_dir = Path(_env("BOWLSENSE_CACHE_DIR") or Path.home() / ".cache" / "bowlsense")
    db_backend = (_env("BOWLSENSE_DB") or "sqlite").lower()
    if db_backend not in ("sqlite", "supabase"):
        raise ConfigError(f"BOWLSENSE_DB must be 'sqlite' or 'supabase', got '{db_backend}'.")
    return EnvSettings(
        config_path=Path(_env("BOWLSENSE_CONFIG") or DEFAULT_CONFIG_PATH),
        db_backend=db_backend,
        sqlite_path=Path(_env("BOWLSENSE_SQLITE_PATH") or cache_dir / "bowlsense.db"),
        cache_dir=cache_dir,
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        supabase_url=_env("SUPABASE_URL"),
        supabase_service_role_key=_env("SUPABASE_SERVICE_ROLE_KEY"),
        web_origin=_env("BOWLSENSE_WEB_ORIGIN") or "http://localhost:3000",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return load_settings()
