"""Run le-wm's *unmodified* ``train.py`` and record its training throughput.

Invoked as ``python -m jmem._official_train_runner <hydra args>`` by
:func:`jmem.train.run_official_train`. ``train.py`` builds its own Lightning
``Trainer`` (``le-wm/train.py:124``), so the only hook available without editing
the script is the ``Trainer`` constructor: we append one callback that
timestamps every optimisation step. Nothing about training is changed.

Environment variables:

* ``JMEM_LEWM_DIR``   path to the le-wm checkout (contains ``train.py``)
* ``JMEM_TRAIN_OUT``  path of the JSON timing record to write
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any


def summarise_step_times(stamps: list[float], warmup_fraction: float = 0.2) -> dict[str, Any]:
    """Median / mean seconds per step from step-end timestamps, skipping warm-up.

    The first ``warmup_fraction`` of the intervals (at least one) is discarded:
    it contains dataloader spin-up and CUDA / autocast warm-up.
    """
    intervals = [b - a for a, b in zip(stamps[:-1], stamps[1:])]
    if not intervals:
        return {"n_intervals": 0, "median_s": None, "mean_s": None}
    skip = min(len(intervals) - 1, max(1, int(len(intervals) * warmup_fraction)))
    kept = intervals[skip:]
    return {
        "n_intervals": len(kept),
        "n_warmup_skipped": skip,
        "median_s": statistics.median(kept),
        "mean_s": statistics.fmean(kept),
        "max_s": max(kept),
    }


def main() -> None:
    lewm_dir = os.environ["JMEM_LEWM_DIR"]
    out_path = os.environ["JMEM_TRAIN_OUT"]
    sys.path.insert(0, lewm_dir)

    import lightning as pl
    import torch
    from lightning.pytorch.callbacks import Callback

    class StepTimer(Callback):
        def __init__(self) -> None:
            self.stamps: list[float] = []
            self.t_fit_start: float | None = None

        def _dump(self, trainer: Any, finished: bool) -> None:
            loader = trainer.train_dataloader
            record = {
                "finished": finished,
                "global_step": int(trainer.global_step),
                "steps_per_epoch": int(trainer.num_training_batches)
                if trainer.num_training_batches not in (None, float("inf"))
                else None,
                "max_epochs": trainer.max_epochs,
                "max_steps": trainer.max_steps,
                "batch_size": getattr(loader, "batch_size", None),
                "num_workers": getattr(loader, "num_workers", None),
                "precision": str(trainer.precision),
                "device": str(trainer.strategy.root_device),
                "step_time": summarise_step_times(self.stamps),
                "fit_wallclock_s": time.perf_counter() - (self.t_fit_start or time.perf_counter()),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            self.t_fit_start = time.perf_counter()

        def on_train_batch_end(self, trainer: Any, pl_module: Any, *args: Any, **kwargs: Any) -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.stamps.append(time.perf_counter())
            if len(self.stamps) % 50 == 0:
                self._dump(trainer, finished=False)

        def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
            self._dump(trainer, finished=True)

    original_init = pl.Trainer.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), StepTimer()]
        original_init(self, *args, **kwargs)

    pl.Trainer.__init__ = patched_init

    import runpy

    train_py = os.path.join(lewm_dir, "train.py")
    sys.argv = [train_py, *sys.argv[1:]]
    runpy.run_path(train_py, run_name="__main__")


if __name__ == "__main__":
    main()
