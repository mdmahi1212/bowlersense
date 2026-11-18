"""Offline guard: when ``BOWLSENSE_OFFLINE=1``, any connection to a non-loopback host raises.

This backs the spec's guarantee that, after the one-time model download, BowlSense works with
the network unplugged. The guard patches the ``socket`` module, so it catches every library
(HTTP clients, model downloaders, telemetry) without needing to know about them.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any

from bowlsense.config import offline_requested
from bowlsense.errors import BowlSenseError

_LOCAL_NAMES: frozenset[str] = frozenset({"localhost", "localhost.localdomain", ""})

_original: dict[str, Any] = {}


class OfflineViolationError(BowlSenseError):
    """Code tried to reach the network while offline mode is on."""

    def __init__(self, host: object) -> None:
        self.host = host
        super().__init__(
            f"Network access to '{host}' blocked: BOWLSENSE_OFFLINE=1 is set and BowlSense "
            "must only talk to localhost. A model or library is trying to download or phone "
            "home; run scripts/download_models.py while online, or disable its telemetry."
        )


def is_local_host(host: object) -> bool:
    """Return True for loopback addresses, ``localhost`` and non-IP socket families."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if not isinstance(host, str):
        return False
    name = host.strip("[]").lower()
    if name in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(name.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _host_of(address: object) -> object:
    if isinstance(address, tuple) and address:
        return address[0]
    # AF_UNIX paths and other non-IP addresses are local by definition.
    return None


def _guard_connect(original: Callable[..., Any]) -> Callable[..., Any]:
    def connect(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not is_local_host(
            _host_of(address)
        ):
            raise OfflineViolationError(_host_of(address))
        return original(self, address)

    return connect


def _guard_getaddrinfo(original: Callable[..., Any]) -> Callable[..., Any]:
    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_local_host(host):
            raise OfflineViolationError(host)
        return original(host, *args, **kwargs)

    return getaddrinfo


def enable() -> None:
    """Install the guard (idempotent)."""
    if _original:
        return
    _original["connect"] = socket.socket.connect
    _original["connect_ex"] = socket.socket.connect_ex
    _original["getaddrinfo"] = socket.getaddrinfo
    socket.socket.connect = _guard_connect(_original["connect"])  # type: ignore[method-assign]
    socket.socket.connect_ex = _guard_connect(  # type: ignore[method-assign]
        _original["connect_ex"]
    )
    socket.getaddrinfo = _guard_getaddrinfo(_original["getaddrinfo"])


def disable() -> None:
    """Remove the guard (used by tests)."""
    if not _original:
        return
    socket.socket.connect = _original.pop("connect")  # type: ignore[method-assign]
    socket.socket.connect_ex = _original.pop("connect_ex")  # type: ignore[method-assign]
    socket.getaddrinfo = _original.pop("getaddrinfo")


def is_enabled() -> bool:
    """Return True while the guard is installed."""
    return bool(_original)


def enable_if_requested() -> bool:
    """Install the guard when ``BOWLSENSE_OFFLINE`` is truthy; return whether it is active."""
    if offline_requested():
        enable()
    return is_enabled()
