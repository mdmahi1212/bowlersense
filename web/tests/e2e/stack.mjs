// Starts a throwaway BowlSense API + worker for the Playwright smoke test.
// Data, config and a rendered synthetic delivery (video + keypoints) go to test-results/e2e-stack.
import { spawn, execFileSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

const web = path.resolve(import.meta.dirname, "..", "..");
const ml = path.resolve(web, "..", "ml");
const root = path.join(web, "test-results", "e2e-stack");
const apiPort = process.env.E2E_API_PORT ?? "8765";
const webPort = process.env.E2E_WEB_PORT ?? "3765";
const python = [
  path.join(ml, ".venv", "Scripts", "python.exe"),
  path.join(ml, ".venv", "bin", "python"),
].find(existsSync);
if (!python) throw new Error("ml/.venv not found: run scripts/setup.py first");

rmSync(root, { recursive: true, force: true });
mkdirSync(root, { recursive: true });
const prep = `
import json, sys, yaml
from pathlib import Path
from bowlsense.config import DEFAULT_CONFIG_PATH
from synth.skeleton_gen import SynthParams, generate, render_to_video
root = Path(sys.argv[1])
r = generate(SynthParams(fps=120, height_px=300, elbow_extension_deg=22.0, knee_flex_change_deg=30.0))
render_to_video(r.sequence, root / "delivery.mp4")
(root / "delivery.pose.json").write_text(r.sequence.model_dump_json(), encoding="utf-8")
cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
cfg["storage"].update(root=str(root / "storage"), db_path=str(root / "db.sqlite"),
                      models_dir=str((DEFAULT_CONFIG_PATH.parents[1] / "models").resolve()))
cfg["jobs"]["poll_interval_s"] = 0.3
(root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
(root / "settings.json").write_text(json.dumps({"coach_model": "llama3.1:8b", "pose_backend": "rtm", "use_llm": False}))
from bowlsense.db.session import make_engine, upgrade
upgrade(make_engine(root / "db.sqlite"))  # migrate once, before the API and worker start
`;
execFileSync(python, ["-c", prep, root], { cwd: ml, stdio: "inherit" });
writeFileSync(path.join(root, "ready"), "1");

const env = {
  ...process.env,
  BOWLSENSE_CONFIG: path.join(root, "config.yaml"),
  BOWLSENSE_API_PORT: apiPort,
  BOWLSENSE_WEB_PORT: webPort,
  BOWLSENSE_OFFLINE: "1",
  PYTHONUNBUFFERED: "1",
};
const children = ["bowlsense.api", "bowlsense.worker"].map((mod) =>
  spawn(python, ["-m", mod], { cwd: ml, env, stdio: "inherit" }),
);
const stop = () => {
  for (const c of children) c.kill();
  process.exit(0);
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
for (const c of children)
  c.on("exit", (code) => code && console.error(`${c.spawnargs.join(" ")} exited ${code}`));
