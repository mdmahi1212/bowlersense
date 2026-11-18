"""Tests for the BOWLSENSE_OFFLINE network guard."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from bowlsense import offline
from bowlsense.offline import OfflineViolationError, is_local_host

ML_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("127.5.5.5", True),
        ("::1", True),
        ("[::1]", True),
        (b"localhost", True),
        (None, True),
        ("example.com", False),
        ("8.8.8.8", False),
        ("192.168.0.10", False),
        (12345, False),
    ],
)
def test_is_local_host(host: object, local: bool) -> None:
    assert is_local_host(host) is local


def test_guard_blocks_remote_allows_local(no_offline_guard: None) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = threading.Thread(target=lambda: server.accept()[0].close())
    accepted.start()
    offline.enable()
    offline.enable()  # idempotent
    try:
        assert offline.is_enabled()
        with pytest.raises(OfflineViolationError, match=r"example.com"):
            socket.getaddrinfo("example.com", 443)
        with pytest.raises(OfflineViolationError, match=r"8\.8\.8\.8"):
            socket.create_connection(("8.8.8.8", 53), timeout=1)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OfflineViolationError):
            s.connect_ex(("1.1.1.1", 80))
        s.close()
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
    finally:
        offline.disable()
        accepted.join(timeout=5)
        server.close()
    assert not offline.is_enabled()
    offline.disable()  # no-op when off


def test_enable_if_requested(monkeypatch: pytest.MonkeyPatch, no_offline_guard: None) -> None:
    monkeypatch.setenv("BOWLSENSE_OFFLINE", "0")
    assert offline.enable_if_requested() is False
    monkeypatch.setenv("BOWLSENSE_OFFLINE", "1")
    assert offline.enable_if_requested() is True


def test_package_import_installs_guard_and_disables_telemetry() -> None:
    code = (
        "import os, socket, bowlsense\n"
        "from bowlsense import offline\n"
        "assert offline.is_enabled()\n"
        "assert os.environ['YOLO_OFFLINE'] == '1'\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 80)\n"
        "except offline.OfflineViolationError:\n"
        "    print('blocked')\n"
    )
    env = {**__import__("os").environ, "BOWLSENSE_OFFLINE": "1"}
    env.pop("YOLO_OFFLINE", None)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ML_DIR, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "blocked"
