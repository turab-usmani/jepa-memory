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
        # the seed the script resolved must be the one that was requested ...
        ("seed", record.get("seed", common["seed"]), hcfg.get("seed")),
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
    # ... and must be one of the protocol's evaluation seeds (42 is the official one;
    # the others are the jmem extension, NOTES D10)
    allowed = protocol.get("eval_seeds") or [common["seed"]]
    if "seed" not in ignore and hcfg.get("seed") not in allowed:
        mismatches.append(f"seed: protocol eval_seeds={allowed!r} runtime={hcfg.get('seed')!r}")
    for name, want, got in checks:
        if name.split(".")[0] in ignore or name in ignore:
            continue
        if want != got:
            mismatches.append(f"{name}: protocol={want!r} runtime={got!r}")
    return mismatches


# --------------------------------------------------------------------------- #
# Results JSON (one file per run x env x perturbation x eval seed)
# --------------------------------------------------------------------------- #

_NUM_OR_NULL = {"type": ["number", "null"]}
_FRACTION = {"type": "number", "minimum": 0.0, "maximum": 1.0}

RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "run_id", "git_sha", "env", "perturbation", "predictor", "params_predictor",
        "train_seed", "eval_seed", "protocol", "success", "n_episodes", "rollout_mse",
        "probe_r2", "latency_ms_median", "hardware", "wallclock_s", "timestamp",
    ],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "git_sha": {"type": "string"},
        "env": {"type": "string"},
        "perturbation": {
            "type": "object", "required": ["name"],
            "properties": {"name": {"type": "string"}},
        },
        "predictor": {"type": "string"},
        "params_predictor": {"type": "integer", "minimum": 0},
        # null = released checkpoint (training seed unknown)
        "train_seed": {"type": ["integer", "null"]},
        "eval_seed": {"type": "integer"},
        "protocol": {"type": "object"},
        # keys are goal offsets in env steps; values are success FRACTIONS in [0, 1]
        "success": {
            "type": "object", "minProperties": 1,
            "propertyNames": {"pattern": "^[0-9]+$"},
            "additionalProperties": _FRACTION,
        },
        "n_episodes": {"type": "integer", "minimum": 1},
        "episode_successes": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "boolean"}},
        },
        # null = not measured in this run (never 0.0 as a placeholder)
        "rollout_mse": {"type": ["object", "null"], "additionalProperties": _NUM_OR_NULL},
        "probe_r2": {"type": ["object", "null"]},
        "latency_ms_median": _NUM_OR_NULL,
        "hardware": {"type": "string"},
        "wallclock_s": {"type": "number", "minimum": 0},
        "timestamp": {"type": "string"},
        # which frames the planner received: "official_1frame" or "stream" (NOTES D9)
        "context": {"type": "string"},
        "versions": {"type": "object"},
    },
}


def success_rate(episode_successes: Sequence[bool]) -> float:
    """Fraction of successful episodes, in [0, 1]."""
    eps = [bool(x) for x in episode_successes]
    if not eps:
        raise ValueError("no episodes")
    return sum(eps) / len(eps)


def validate_result(result: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` / ``ValueError`` if ``result`` is malformed."""
    import math

    import jsonschema

    jsonschema.validate(result, RESULT_SCHEMA)
    for off, eps in (result.get("episode_successes") or {}).items():
        if len(eps) != result["n_episodes"]:
            raise ValueError(f"offset {off}: {len(eps)} episodes, expected {result['n_episodes']}")
        if not math.isclose(success_rate(eps), result["success"][off], abs_tol=1e-9):
            raise ValueError(f"offset {off}: success does not match episode_successes")


def official_record_to_result(record: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Convert a record from :func:`run_official_eval` into the results-JSON schema.

    Only planning success is filled in; rollout error, probes and latency are
    ``None`` because the official script does not measure them.
    """
    env = record["env"]
    offset = str(record["evaluate_kwargs"]["goal_offset"])
    eps = [bool(x) for x in record["metrics"]["episode_successes"]]
    model = record["policy"].get("model") or {}
    frames = model.get("predictor_num_frames")
    result = {
        "run_id": f"official-{env}-released-s{record['seed']}",
        "git_sha": record["git_sha"],
        "env": protocol["envs"][env]["env_name"],
        "perturbation": {"name": "none"},
        "predictor": f"transformer_h{frames}" if frames else record["policy"]["type"],
        "params_predictor": int(model.get("n_params_predictor") or 0),
        "train_seed": None,
        "eval_seed": int(record["seed"]),
        "protocol": {
            "common": protocol["common"],
            "env": protocol["envs"][env],
            "source": protocol["source"],
            "mismatches": check_protocol(record, protocol, env),
        },
        "success": {offset: success_rate(eps)},
        "n_episodes": len(eps),
        "episode_successes": {offset: eps},
        "rollout_mse": None,
        "probe_r2": None,
        "latency_ms_median": None,
        "hardware": record["hardware"],
        "wallclock_s": float(record["wallclock_s"]),
        "timestamp": record["timestamp"],
        "context": "official_1frame",
        "versions": record.get("versions", {}),
    }
    validate_result(result)
    return result


def write_result(result: dict[str, Any], out_dir: Path | str) -> Path:
    """Validate and write ``<out_dir>/<run_id>.json``."""
    validate_result(result)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
