"""Compute ledger: an append-only CSV of GPU time, checked against the budget.

Usage in a notebook::

    from jmem import ledger
    ledger.start("nb00-smoke", est_hours=0.3, notebook="00_setup_and_smoke")
    ...
    ledger.stop()
    ledger.summary()

A row is written at ``start`` (with an empty ``end``) and completed at ``stop``.
If a Colab session dies before ``stop``, the row stays *open*; ``summary``
charges open rows at their ``est_hours`` (never less than the time already
elapsed is knowable, so the estimate is the conservative stand-in) and lists
them so they can be closed by hand with :func:`close_open`.

The budget is expressed in L4-hours. Hours on other GPUs are converted with
``L4_EQUIV`` (150 L4-h == 60 A100-h, i.e. 1 A100-h == 2.5 L4-h). CPU time is free.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["run_id", "notebook", "gpu", "start", "end", "hours", "est_hours"]
BUDGET_L4_HOURS = 150.0
L4_EQUIV = {"L4": 1.0, "A100": 2.5, "cpu": 0.0}
# Unknown GPUs (e.g. T4) are charged 1:1 so that they are never under-counted.
DEFAULT_EQUIV = 1.0

_active: dict[str, str] | None = None
_active_path: Path | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _default_path() -> Path:
    from jmem.paths import get_paths

    return get_paths().ledger_csv


def read(path: Path | None = None) -> list[dict[str, str]]:
    """Return all ledger rows (empty list if the file does not exist)."""
    path = Path(path) if path is not None else _default_path()
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_all(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _append(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def start(
    run_id: str,
    est_hours: float,
    notebook: str = "",
    gpu: str | None = None,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    """Open a ledger entry and append it to the CSV.

    Args:
        run_id: Unique label for this block of compute.
        est_hours: Estimated wall-clock hours (stated before the run starts).
        notebook: Notebook or script name.
        gpu: GPU label (``"L4"``, ``"A100"``, ``"cpu"``...). Auto-detected if None.
        path: Ledger CSV (defaults to ``paths.ledger_csv``).
        now: Clock override for tests.
    """
    global _active, _active_path
    if _active is not None:
        raise RuntimeError(
            f"ledger entry {_active['run_id']!r} is still open; call ledger.stop() first"
        )
    if est_hours < 0:
        raise ValueError("est_hours must be >= 0")
    if gpu is None:
        from jmem.utils import detect_gpu

        gpu = detect_gpu()
    path = Path(path) if path is not None else _default_path()
    row = {
        "run_id": run_id,
        "notebook": notebook,
        "gpu": gpu,
        "start": _fmt(now or _now()),
        "end": "",
        "hours": "",
        "est_hours": f"{float(est_hours):.4f}",
    }
    _append(path, row)
    _active, _active_path = row, path
    return dict(row)


def _close(row: dict[str, str], end: datetime) -> None:
    hours = (end - _parse(row["start"])).total_seconds() / 3600.0
    if hours < 0:
        raise ValueError("ledger entry ends before it starts")
    row["end"] = _fmt(end)
    row["hours"] = f"{hours:.4f}"


def stop(now: datetime | None = None) -> dict[str, str]:
    """Close the entry opened by :func:`start` and write its duration."""
    global _active, _active_path
    if _active is None or _active_path is None:
        raise RuntimeError("no open ledger entry; call ledger.start() first")
    rows = read(_active_path)
    for row in reversed(rows):
        if row["run_id"] == _active["run_id"] and row["start"] == _active["start"]:
            _close(row, now or _now())
            closed = dict(row)
            break
    else:
        raise RuntimeError("open ledger entry not found in CSV (file edited?)")
    _write_all(_active_path, rows)
    _active, _active_path = None, None
    return closed


def close_open(
    run_id: str, hours: float, path: Path | None = None
) -> dict[str, str]:
    """Manually close a row left open by a crashed session, charging ``hours``."""
    path = Path(path) if path is not None else _default_path()
    rows = read(path)
    for row in rows:
        if row["run_id"] == run_id and not row["end"]:
            row["end"] = "manual"
            row["hours"] = f"{float(hours):.4f}"
            _write_all(path, rows)
            return dict(row)
    raise KeyError(f"no open ledger row with run_id={run_id!r}")


def l4_equivalent(hours: float, gpu: str) -> float:
    """Convert ``hours`` on ``gpu`` into L4-hours."""
    return hours * L4_EQUIV.get(gpu, DEFAULT_EQUIV)


@dataclass(frozen=True)
class Summary:
    """Budget position, all in L4-equivalent hours."""

    used: float
    open_est: float
    planned: float
    projected: float
    budget: float
    over_budget: bool
    open_runs: tuple[str, ...]

    @property
    def remaining(self) -> float:
        return self.budget - self.projected

    @property
    def fraction_used(self) -> float:
        return self.used / self.budget if self.budget else float("inf")


def summary(
    path: Path | None = None,
    budget: float = BUDGET_L4_HOURS,
    planned_hours: float = 0.0,
    planned_gpu: str = "L4",
    verbose: bool = True,
) -> Summary:
    """Summarise hours used against the budget.

    ``projected = used + est. hours of open rows + planned``. A warning line is
    printed when the projection exceeds the budget.

    Args:
        path: Ledger CSV (defaults to ``paths.ledger_csv``).
        budget: Budget in L4-hours.
        planned_hours: Hours of a run being proposed (to check before launching).
        planned_gpu: GPU the proposed run would use.
        verbose: Print a human-readable summary.
    """
    used = 0.0
    open_est = 0.0
    open_runs: list[str] = []
    for row in read(path):
        if row["end"]:
            used += l4_equivalent(float(row["hours"]), row["gpu"])
        else:
            open_est += l4_equivalent(float(row["est_hours"]), row["gpu"])
            open_runs.append(row["run_id"])
    planned = l4_equivalent(planned_hours, planned_gpu)
    projected = used + open_est + planned
    s = Summary(
        used=used,
        open_est=open_est,
        planned=planned,
        projected=projected,
        budget=budget,
        over_budget=projected > budget,
        open_runs=tuple(open_runs),
    )
    if verbose:
        print(
            f"[ledger] used {s.used:.2f} / {s.budget:.0f} L4-h "
            f"({100 * s.fraction_used:.1f}%); open (est.) {s.open_est:.2f}; "
            f"planned {s.planned:.2f}; projected {s.projected:.2f}; "
            f"remaining {s.remaining:.2f}"
        )
        if s.open_runs:
            print(f"[ledger] open rows charged at their estimate: {list(s.open_runs)}")
        if s.over_budget:
            print(
                f"[ledger] WARNING: projected {s.projected:.2f} L4-h exceeds the "
                f"{s.budget:.0f} L4-h budget by {-s.remaining:.2f}"
            )
    return s
