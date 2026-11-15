"""Seeding, config resolution, parameter counting and environment capture."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "stable-worldmodel",
    "stable-pretraining",
    "transformers",
    "lightning",
    "hydra-core",
    "omegaconf",
    "gymnasium",
    "scikit-learn",
    "h5py",
    "hdf5plugin",
    "pymunk",
    "pygame",
    "mujoco",
    "dm_control",
    "ogbench",
)


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (CPU and CUDA). Returns the seed."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def count_params(module: Any, trainable_only: bool = False) -> int:
    """Number of parameters in a ``torch.nn.Module`` (each tensor counted once)."""
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def classify_gpu(name: str) -> str:
    """Map a device name (``torch.cuda.get_device_name``) to a ledger label."""
    upper = name.upper()
    for label in ("A100", "L4", "T4", "V100", "H100", "P100"):
        if label in upper.replace("-", " ").split() or f" {label}" in f" {upper}":
            return label
    return name.strip() or "unknown"


def detect_gpu() -> str:
    """Return ``"L4"``, ``"A100"``, another GPU name, or ``"cpu"``."""
    try:
        import torch

        if torch.cuda.is_available():
            return classify_gpu(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def git_sha(repo: Path | None = None) -> str:
    """Current commit hash of ``repo`` (suffix ``-dirty`` if there are local changes)."""
    from jmem.paths import repo_root

    repo = repo or repo_root()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def env_versions() -> dict[str, str]:
    """Installed versions of the packages that affect results."""
    out = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp (seconds resolution)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(*configs: dict[str, Any] | Path | str) -> dict[str, Any]:
    """Merge YAML files / dicts left to right into one resolved config."""
    resolved: dict[str, Any] = {}
    for c in configs:
        resolved = deep_update(resolved, c if isinstance(c, dict) else load_yaml(c))
    return resolved


def save_resolved_config(config: dict[str, Any], out_dir: Path | str, name: str = "config") -> Path:
    """Write the full resolved config (plus provenance) next to a run's results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "git_sha": git_sha(),
        "versions": env_versions(),
        "hardware": detect_gpu(),
        "timestamp": utc_timestamp(),
    }
    path = out_dir / f"{name}.resolved.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
