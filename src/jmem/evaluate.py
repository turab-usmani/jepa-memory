"""Evaluation entrypoints.

Phase 0 scope: run the official LeWM evaluation (``le-wm/eval.py``) unchanged,
capture what it did at runtime, and compare that against
``configs/eval/protocol.yaml``. The Phase-1 metrics (success@offset, rollout
error, probes, latency, results-JSON schema) are added to this module later.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from jmem.paths import Paths, get_paths, repo_root
from jmem.utils import load_yaml

PROTOCOL_PATH = repo_root() / "configs" / "eval" / "protocol.yaml"


def load_protocol(path: Path | str | None = None) -> dict[str, Any]:
    """Load THE evaluation protocol (``configs/eval/protocol.yaml``)."""
    return load_yaml(path or PROTOCOL_PATH)


def official_eval_command(
    env: str,
    policy: str,
    num_eval: int,
    seed: int,
    data_home: Path | str,
    hydra_dir: Path | str,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build the command line that runs le-wm's eval.py through our capture runner."""
    return [
        sys.executable,
        "-m",
        "jmem._official_eval_runner",
        f"--config-name={env}",
        f"policy={policy}",
        f"eval.num_eval={int(num_eval)}",
        f"seed={int(seed)}",
        f"cache_dir={Path(data_home).as_posix()}",
        f"hydra.run.dir={Path(hydra_dir).as_posix()}",
        *extra_overrides,
    ]


def run_official_eval(
    env: str,
    policy: str,
    out_dir: Path | str,
    num_eval: int = 50,
    seed: int = 42,
    paths: Paths | None = None,
    extra_overrides: Sequence[str] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Run the official LeWM evaluation for one env and return the captured record.

    The record (also written to ``<out_dir>/official_<env>_n<num_eval>_s<seed>.json``)
    contains the runtime arguments, planner settings, metrics and the Hydra
    config the script resolved. Existing records are reused unless ``force``.

    Args:
        env: le-wm eval config name: ``tworoom``, ``pusht``, ``reacher`` or ``cube``.
        policy: Checkpoint name understood by ``swm.wm.utils.load_pretrained``
            (e.g. ``quentinll/lewm-tworooms``) or ``random``.
        out_dir: Directory for the record and Hydra output.
        num_eval: Number of evaluation episodes (official: 50).
        seed: Evaluation seed (official: 42). Selects episodes and seeds CEM.
        paths: Project paths (defaults to :func:`jmem.paths.get_paths`).
        extra_overrides: Additional Hydra overrides, passed through verbatim.
        force: Re-run even if the record exists.
    """
    paths = paths or get_paths()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"official_{env}_n{num_eval}_s{seed}"
    record_path = out_dir / f"{tag}.json"
    if record_path.exists() and not force:
        print(f"[eval] reusing {record_path}")
        return json.loads(record_path.read_text(encoding="utf-8"))

    hydra_dir = out_dir / f"{tag}_hydra"
    raw_path = out_dir / f"{tag}.raw.json"
    raw_path.unlink(missing_ok=True)
    cmd = official_eval_command(
        env, policy, num_eval, seed, paths.data_home, hydra_dir, extra_overrides
    )
    child_env = dict(os.environ)
    child_env["JMEM_LEWM_DIR"] = str(paths.lewm_dir)
    child_env["JMEM_EVAL_OUT"] = str(raw_path)
    child_env["STABLEWM_HOME"] = str(paths.swm_home)
    print("[eval] " + " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=child_env, cwd=str(paths.lewm_dir))
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"official eval failed with exit code {proc.returncode}")
    if not raw_path.exists():
        raise RuntimeError("official eval finished without calling World.evaluate")

    record = json.loads(raw_path.read_text(encoding="utf-8"))
    hydra_cfg = hydra_dir / ".hydra" / "config.yaml"
    record["hydra_config"] = load_yaml(hydra_cfg) if hydra_cfg.exists() else None
    record["env"] = env
    record["policy_name"] = policy
    record["seed"] = seed
    record["wallclock_s"] = wall
    record["command"] = cmd
    from jmem.utils import detect_gpu, env_versions, git_sha, utc_timestamp

    record["git_sha"] = git_sha()
    record["versions"] = env_versions()
    record["hardware"] = detect_gpu()
    record["timestamp"] = utc_timestamp()
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    raw_path.unlink(missing_ok=True)
    return record


def check_protocol(
    record: dict[str, Any],
    protocol: dict[str, Any],
    env: str,
    ignore: Sequence[str] = (),
) -> list[str]:
    """Compare a runtime record against the protocol; return human-readable mismatches.

    An empty list means the run used exactly the protocol's values. ``ignore``
    lists protocol keys that were deliberately overridden (e.g. ``num_eval`` in
    the 2-episode smoke test).
    """
    common = protocol["common"]
    spec = protocol["envs"][env]
    kwargs = record["evaluate_kwargs"]
    policy = record["policy"]
    plan = policy.get("plan_config", {})
    solver = policy.get("solver", {})
    hcfg = record.get("hydra_config") or {}

    checks: list[tuple[str, Any, Any]] = [
        ("goal_offset_steps", common["goal_offset_steps"], kwargs.get("goal_offset")),
        ("eval_budget", common["eval_budget"], kwargs.get("eval_budget")),
        ("num_eval", common["num_eval"], record.get("num_envs")),
        ("num_eval", common["num_eval"], len(kwargs.get("episodes_idx") or [])),
        ("seed", common["seed"], hcfg.get("seed")),
        ("img_size", common["img_size"], (hcfg.get("eval") or {}).get("img_size")),
        ("max_episode_steps", common["max_episode_steps"], record.get("max_episode_steps")),
        ("callables", spec["callables"], kwargs.get("callables")),
        ("env_name", spec["env_name"], (hcfg.get("world") or {}).get("env_name")),
        ("dataset_name", spec["dataset_name"], (hcfg.get("eval") or {}).get("dataset_name")),
        (
            "context_frames_at_plan_time",
            common["context_frames_at_plan_time"],
            (record.get("info_pixels_shape") or [None, None])[1],
        ),
    ]
    for key, want in common["plan"].items():
        checks.append((f"plan.{key}", want, plan.get(key)))
    for key, want in common["solver"].items():
        got = solver.get(key)
        if key == "target":  # compare class names; the runtime reports the defining module
            want = str(want).rsplit(".", 1)[-1]
            got = str(solver.get("type", "")).rsplit(".", 1)[-1]
        checks.append((f"solver.{key}", want, got))

    mismatches = []
    for name, want, got in checks:
        if name.split(".")[0] in ignore or name in ignore:
            continue
        if want != got:
            mismatches.append(f"{name}: protocol={want!r} runtime={got!r}")
    return mismatches
