"""Run le-wm's *unmodified* ``eval.py`` and capture what it actually did.

Invoked as ``python -m jmem._official_eval_runner <hydra args>`` by
:func:`jmem.evaluate.run_official_eval`. The official script only prints its
metrics, so we wrap ``swm.World.evaluate`` (the single call at
``le-wm/eval.py:144``) to dump, as JSON:

* the arguments the script passed (goal offset, budget, start steps, episodes);
* the planner / solver settings living on the policy object at that moment;
* the returned metrics.

Nothing about the evaluation itself is changed. Environment variables:

* ``JMEM_LEWM_DIR``  path to the le-wm checkout (contains ``eval.py``)
* ``JMEM_EVAL_OUT``  path of the JSON file to write
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from typing import Any


def _to_jsonable(x: Any) -> Any:
    import numpy as np

    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _describe_policy(policy: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(policy).__name__}
    cfg = getattr(policy, "cfg", None)
    if dataclasses.is_dataclass(cfg):
        out["plan_config"] = dataclasses.asdict(cfg)
    solver = getattr(policy, "solver", None)
    if solver is not None:
        out["solver"] = {"type": f"{type(solver).__module__}.{type(solver).__name__}"}
        for attr in ("num_samples", "n_steps", "topk", "var_scale", "batch_size"):
            if hasattr(solver, attr):
                out["solver"][attr] = getattr(solver, attr)
        model = getattr(solver, "model", None)
        if model is not None:
            out["model"] = {
                "type": f"{type(model).__module__}.{type(model).__name__}",
                "predictor_num_frames": getattr(
                    getattr(model, "predictor", None), "num_frames", None
                ),
                "n_params": sum(p.numel() for p in model.parameters()),
                "n_params_predictor": sum(
                    p.numel() for p in getattr(model, "predictor").parameters()
                )
                if hasattr(model, "predictor")
                else None,
            }
    process = getattr(policy, "process", None)
    if process:
        out["process_keys"] = sorted(process.keys())
    return out


def main() -> None:
    lewm_dir = os.environ["JMEM_LEWM_DIR"]
    out_path = os.environ["JMEM_EVAL_OUT"]
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # pygame envs, headless
    sys.path.insert(0, lewm_dir)

    import stable_worldmodel as swm

    original = swm.World.evaluate

    def evaluate(self, *args: Any, **kwargs: Any) -> dict:
        t0 = time.perf_counter()
        metrics = original(self, *args, **kwargs)
        wall = time.perf_counter() - t0
        pixels = self.infos.get("pixels")
        record = {
            "evaluate_kwargs": {
                k: _to_jsonable(v)
                for k, v in kwargs.items()
                if k not in ("dataset", "video")
            },
            "n_positional_args": len(args),
            "num_envs": self.num_envs,
            # set in code (eval.py:57), so it is absent from Hydra's saved config
            "max_episode_steps": getattr(
                getattr(self.envs.envs[0], "spec", None), "max_episode_steps", None
            ),
            # (num_envs, T, H, W, C): T is the number of real frames the planner sees
            "info_pixels_shape": list(pixels.shape) if pixels is not None else None,
            "policy": _describe_policy(self.policy),
            "metrics": _to_jsonable(metrics),
            "evaluate_wallclock_s": wall,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return metrics

    swm.World.evaluate = evaluate

    # Execute le-wm/eval.py exactly as `python eval.py ...` would: Hydra resolves
    # config_path="./config/eval" relative to the file only when it is __main__.
    import runpy

    eval_py = os.path.join(lewm_dir, "eval.py")
    sys.argv = [eval_py, *sys.argv[1:]]
    runpy.run_path(eval_py, run_name="__main__")


if __name__ == "__main__":
    main()
