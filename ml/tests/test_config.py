"""Tests for the YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bowlsense.config import (
    DEFAULT_CONFIG_PATH,
    FAULT_CODES,
    Settings,
    get_settings,
    load_env,
    load_settings,
)
from bowlsense.errors import ConfigError


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _raw() -> dict:  # type: ignore[type-arg]
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_default_config_loads(settings: Settings) -> None:
    assert settings.legality.elbow_extension_threshold_deg == 15.0
    assert settings.pose.backend == "rtm"
    assert set(settings.faults.weights) == set(FAULT_CODES)
    assert abs(sum(settings.injury.weights.values()) - 1.0) < 1e-9
    assert settings.coach.model == "claude-sonnet-5-5"
    assert settings.jobs.max_attempts == 3


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("video: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_settings(p)


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(_write(tmp_path, [1, 2]))


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("injury", "weights", {"lumbar": 1.0}, "keys"),
        ("faults", "understride_pct", 95, "understride"),
        ("pose", "backend", "openpose", "backend"),
        ("video", "min_fps_recommended", 30, "min_fps_recommended"),
        (
            "delivery",
            "length_zones_m",
            {"yorker": 6, "full": 2, "good": 8, "short": 10},
            "increasing",
        ),
        ("api", "allowed_extensions", ["mp4"], "allowed_extensions"),
    ],
)
def test_validation_errors(
    tmp_path: Path, section: str, key: str, value: object, match: str
) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ConfigError, match=match):
        load_settings(_write(tmp_path, raw))


def test_injury_weights_must_sum_to_one(tmp_path: Path) -> None:
    raw = _raw()
    raw["injury"]["weights"]["lumbar"] = 0.9
    with pytest.raises(ConfigError, match="sum to 1"):
        load_settings(_write(tmp_path, raw))


def test_unknown_key_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["video"]["typo_key"] = 1
    with pytest.raises(ConfigError):
        load_settings(_write(tmp_path, raw))


def test_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "BOWLSENSE_DB",
        "BOWLSENSE_CONFIG",
        "BOWLSENSE_SQLITE_PATH",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BOWLSENSE_CACHE_DIR", str(tmp_path))
    env = load_env()
    assert env.db_backend == "sqlite"
    assert env.sqlite_path == tmp_path / "bowlsense.db"
    assert env.config_path == DEFAULT_CONFIG_PATH
    assert env.anthropic_api_key is None


def test_env_bad_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOWLSENSE_DB", "mysql")
    with pytest.raises(ConfigError, match="BOWLSENSE_DB"):
        load_env()


def test_env_config_path_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOWLSENSE_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigError, match=r"missing.yaml"):
        load_settings()
