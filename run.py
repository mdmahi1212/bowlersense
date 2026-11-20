"""Start BowlSense locally with one command: ``python run.py``.

Starts the API (127.0.0.1), the job worker and the web app, checks the model files and the
local Ollama server, then opens the browser. Ctrl+C stops everything. Works on Windows, Linux
and macOS; run ``python scripts/setup.py`` once first.

Options: ``--no-browser``, ``--dev`` (Next.js dev server), ``--offline`` (block every
non-localhost connection), ``--api-port``/``--web-port`` (defaults from config/default.yaml;
the next free port is used if one is busy).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ML = ROOT / "ml"
WEB = ROOT / "web"
HOST = "127.0.0.1"


def venv_python() -> Path | None:
    """The project's virtualenv interpreter, if set up."""
    for p in (ML / ".venv" / "Scripts" / "python.exe", ML / ".venv" / "bin" / "python"):
        if p.is_file():
            return p
    return None


def ensure_venv() -> None:
    """Re-run this script with the project's virtualenv Python (it has the dependencies)."""
    py = venv_python()
    if py is None:
        sys.exit("BowlSense isn't set up yet. Run:  python scripts/setup.py")
    if Path(sys.executable).resolve() != py.resolve() and not os.environ.get("BOWLSENSE_RUN_CHILD"):
        env = {**os.environ, "BOWLSENSE_RUN_CHILD": "1"}
        sys.exit(subprocess.call([str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


def port_free(port: int) -> bool:
    """True if nothing is listening on 127.0.0.1:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


def pick_port(wanted: int, taken: set[int], label: str) -> int:
    """``wanted`` if free, else the next free port (with a clear message)."""
    port = wanted
    while port in taken or not port_free(port):
        port += 1
        if port > wanted + 100:
            sys.exit(f"No free port found for the {label} near {wanted}.")
    if port != wanted:
        print(f"  Port {wanted} is in use; the {label} will use port {port} instead.")
    return port


def wait_http(url: str, timeout_s: float, procs: list[subprocess.Popen[bytes]]) -> bool:
    """Poll ``url`` until it answers (or a process dies, or the timeout passes)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in procs):
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def npm() -> str:
    """The npm executable (``npm.cmd`` on Windows)."""
    exe = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which("npm")
    if exe is None:
        sys.exit("npm wasn't found. Install Node.js 20+ and run python scripts/setup.py.")
    return exe


def check_models(settings: object) -> None:
    """Print which model files are present."""
    from bowlsense.config import Settings

    assert isinstance(settings, Settings)
    folder = settings.storage.models_path
    needed = [settings.detect.yolo_model, settings.pose.rtm_model, settings.pose.mediapipe_model]
    missing = [n for n in needed if not (folder / n).is_file()]
    if missing:
        print(f"  Missing model files in {folder}: {', '.join(missing)}")
        print("  Run once (needs the internet):  python scripts/download_models.py")
    else:
        print("  Model files: OK")


def check_ollama(settings: object) -> None:
    """Report whether the local LLM coach is available (the template coach is used if not)."""
    from bowlsense.coach.llm import OllamaClient, OllamaUnavailableError
    from bowlsense.config import Settings
    from bowlsense.local_settings import effective_settings, load_local_settings

    assert isinstance(settings, Settings)
    cfg = effective_settings(settings, load_local_settings(settings)).coach
    client = OllamaClient(cfg.model_copy(update={"timeout_s": 3}))
    try:
        if client.has_model(cfg.model):
            print(f"  Ollama: running, model {cfg.model} ready")
        else:
            print(f"  Ollama: running, but {cfg.model} isn't pulled (ollama pull {cfg.model}).")
            print("          The rule-based coach report is used until then.")
    except OllamaUnavailableError:
        print("  Ollama: not running. The rule-based coach report is used (start Ollama for")
        print("          the AI coach; installing Ollama is a one-time step, see README).")
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    """Start the API, worker and web app; stop them all on Ctrl+C or if one exits."""
    ensure_venv()
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)  # status lines appear immediately, even in a log file
    sys.path.insert(0, str(ML))
    parser = argparse.ArgumentParser(description="Run BowlSense locally")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--dev", action="store_true", help="Next.js dev server (hot reload)")
    parser.add_argument("--offline", action="store_true", help="block non-localhost network")
    parser.add_argument("--api-port", type=int, help="API port (default from config)")
    parser.add_argument("--web-port", type=int, help="web port (default from config)")
    args = parser.parse_args(argv)

    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    if args.offline:
        os.environ["BOWLSENSE_OFFLINE"] = "1"
    from bowlsense.config import load_settings
    from bowlsense.db.session import get_engine

    settings = load_settings()
    print("BowlSense: starting (everything stays on this computer)")
    api_port = pick_port(args.api_port or settings.server.api_port, set(), "API")
    web_port = pick_port(args.web_port or settings.server.web_port, {api_port}, "web app")
    get_engine(str(settings.storage.db_file))  # migrate once, before the API and worker start
    print(f"  Database: {settings.storage.db_file}")
    check_models(settings)
    check_ollama(settings)

    env = {
        **os.environ,
        "BOWLSENSE_API_PORT": str(api_port),
        "BOWLSENSE_WEB_PORT": str(web_port),
        "BOWLSENSE_API_URL": f"http://{HOST}:{api_port}",
        "PYTHONUNBUFFERED": "1",
    }
    if not args.dev and not (WEB / ".next" / "BUILD_ID").is_file():
        print("  Building the web app (first run only)…")
        if subprocess.call([npm(), "run", "build"], cwd=WEB, env=env) != 0:
            sys.exit("The web app build failed; see the output above.")
    py = sys.executable
    web_cmd = [npm(), "run", "dev" if args.dev else "start", "--", "-p", str(web_port)]
    procs: list[subprocess.Popen[bytes]] = []
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    for cmd, cwd in (
        ([py, "-m", "bowlsense.api"], ML),
        ([py, "-m", "bowlsense.worker"], ML),
        (web_cmd, WEB),
    ):
        procs.append(subprocess.Popen(cmd, cwd=cwd, env=env, creationflags=creation))

    def stop(*_: object) -> None:
        for p in procs:
            if p.poll() is None:
                if os.name == "nt":
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    def on_sigterm(*_: object) -> None:
        stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)
    url = f"http://{HOST}:{web_port}"
    try:
        if not wait_http(f"http://{HOST}:{api_port}/health", 60, procs):
            print("The API didn't start; see the messages above.")
            stop()
            return 1
        if not wait_http(url, 120, procs):
            print("The web app didn't start; see the messages above.")
            stop()
            return 1
        print(f"\nBowlSense is running: {url}   (API {HOST}:{api_port})  Press Ctrl+C to stop.\n")
        if not args.no_browser:
            webbrowser.open(url)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        dead = next(p for p in procs if p.poll() is not None)
        print(f"A BowlSense process exited ({dead.args!r}); stopping.")
        stop()
        return 1
    except KeyboardInterrupt:
        print("\nStopping BowlSense…")
        stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
