"""Training entrypoints.

Phase 1 scope: run the official LeWM training script (``le-wm/train.py``)
unchanged, for (a) a short throughput probe that turns "how long is one run?"
into a measured number before any long run is proposed, and (b) the single
from-scratch reproduction run. Predictor variants (Phase 2) are added later.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from jmem.paths import Paths, get_paths


def official_train_command(
    data: str,
    subdir: str,
    hydra_dir: Path | str,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Command line running le-wm's train.py through our timing runner.

    ``subdir`` must be given: the default ``${hydra:job.id}`` only resolves under
    a cluster launcher. Checkpoints go to ``$STABLEWM_HOME/checkpoints/<subdir>/``.
    """
    return [
        sys.executable,
        "-m",
        "jmem._official_train_runner",
        f"data={data}",
        f"subdir={subdir}",
        f"output_model_name={subdir}",
        f"hydra.run.dir={Path(hydra_dir).as_posix()}",
        *extra_overrides,
    ]


def run_official_train(
    data: str,
    subdir: str,
    out_dir: Path | str,
    paths: Paths | None = None,
    extra_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the official LeWM training script and return its timing record.

    Args:
        data: le-wm data config name (``tworoom``, ``pusht``, ``dmc``, ``ogb``).
        subdir: Run name (checkpoint sub-directory and model name).
        out_dir: Directory for the timing record and Hydra output.
        paths: Project paths (defaults to :func:`jmem.paths.get_paths`).
        extra_overrides: Hydra overrides, e.g. ``["+trainer.max_steps=300"]``.
    """
    paths = paths or get_paths()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_path = out_dir / f"train_{subdir}.timing.json"
    cmd = official_train_command(data, subdir, out_dir / f"train_{subdir}_hydra", extra_overrides)
    child_env = dict(os.environ)
    child_env["JMEM_LEWM_DIR"] = str(paths.lewm_dir)
    child_env["JMEM_TRAIN_OUT"] = str(timing_path)
    child_env["STABLEWM_HOME"] = str(paths.swm_home)
    child_env["LOCAL_DATASET_DIR"] = str(paths.data_home)  # le-wm/train.py:55
    print("[train] " + " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=child_env, cwd=str(paths.lewm_dir))
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"official training failed with exit code {proc.returncode}")
    if not timing_path.exists():
        raise RuntimeError("training finished without producing a timing record")
    record = json.loads(timing_path.read_text(encoding="utf-8"))
    from jmem.utils import detect_gpu, env_versions, git_sha, utc_timestamp

    record.update(
        data=data, subdir=subdir, command=cmd, wallclock_s=wall, git_sha=git_sha(),
        versions=env_versions(), hardware=detect_gpu(), timestamp=utc_timestamp(),
    )
    timing_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def estimate_run_hours(record: dict[str, Any], epochs: int, steps_per_epoch: int) -> dict[str, float]:
    """Extrapolate a full run from a throughput probe.

    Uses the median step time (warm-up excluded). Validation passes, per-epoch
    checkpoint writes and dataloader restarts are NOT included, so this is a
    lower bound; the caller should state that.
    """
    median = record["step_time"]["median_s"]
    if median is None:
        raise ValueError("timing record has no step intervals")
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    total_steps = epochs * steps_per_epoch
    return {
        "median_step_s": median,
        "steps_per_epoch": float(steps_per_epoch),
        "total_steps": float(total_steps),
        "hours_per_epoch": median * steps_per_epoch / 3600.0,
        "hours_total": median * total_steps / 3600.0,
    }


def clips_per_epoch(
    ep_lengths: Sequence[int], num_steps: int, frameskip: int, train_split: float, batch_size: int
) -> int:
    """Optimisation steps per epoch implied by the dataset and le-wm's loader settings.

    Mirrors ``swm/data/dataset.py:48-55`` (one clip per valid start index, span =
    num_steps * frameskip), ``le-wm/train.py:74-78`` (random split of clips,
    ``drop_last=True``).
    """
    span = num_steps * frameskip
    n_clips = sum(max(0, int(length) - span + 1) for length in ep_lengths)
    n_train = int(round(train_split * n_clips))
    return n_train // batch_size
