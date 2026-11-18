"""Tests for the YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bowlsense.config import (
    DEFAULT_CONFIG_PATH,
    FAULT_CODES,
    REPO_ROOT,
    Settings,
    get_settings,
    is_local_url,
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
    assert settings.coach.model == "llama3.1:8b"
    assert settings.coach.timeout_s == 120
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


def test_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOWLSENSE_CONFIG", raising=False)
    monkeypatch.delenv("BOWLSENSE_OFFLINE", raising=False)
    env = load_env()
    assert env.config_path == DEFAULT_CONFIG_PATH
    assert env.offline is False


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("0", False)])
def test_env_offline_flag(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("BOWLSENSE_OFFLINE", value)
    assert load_env().offline is expected


def test_env_config_path_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOWLSENSE_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigError, match=r"missing\.yaml"):
        load_settings()


def test_local_only_defaults(settings: Settings) -> None:
    assert settings.coach.provider == "ollama"
    assert is_local_url(settings.coach.ollama_url)
    assert settings.server.host == "127.0.0.1"
    assert settings.server.web_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_storage_paths_resolve_to_repo(settings: Settings) -> None:
    assert settings.storage.db_file == REPO_ROOT / "data" / "bowlsense.db"
    assert settings.storage.root_path == REPO_ROOT / "data" / "storage"
    assert settings.storage.models_path == REPO_ROOT / "models"


def test_absolute_storage_path_kept(tmp_path: Path) -> None:
    raw = _raw()
    raw["storage"]["db_path"] = str(tmp_path / "x.db")
    assert load_settings(_write(tmp_path, raw)).storage.db_file == tmp_path / "x.db"


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("https://api.example.com", False),
        ("http://192.168.1.5:11434", False),
        ("ftp://localhost", False),
    ],
)
def test_is_local_url(url: str, ok: bool) -> None:
    assert is_local_url(url) is ok


def test_remote_ollama_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["coach"]["ollama_url"] = "https://ollama.example.com"
    with pytest.raises(ConfigError, match="localhost"):
        load_settings(_write(tmp_path, raw))


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [("host", "0.0.0.0", "loopback"), ("web_port", 8000, "differ")],
)
def test_server_validation(tmp_path: Path, key: str, value: object, match: str) -> None:
    raw = _raw()
    raw["server"][key] = value
    with pytest.raises(ConfigError, match=match):
        load_settings(_write(tmp_path, raw))
