"""Results JSON -> tables (mean and 95% CI).

Two intervals are reported and always labelled:

* ``ci_seed``    two-sided 95% Student-t interval of the mean across seeds
  (between-seed spread; with 3 seeds the t quantile is 4.303, so it is wide);
* ``ci_wilson``  95% Wilson score interval on the pooled episodes (binomial
  sampling error only; ignores that episodes within a seed share CEM noise).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# two-sided 95% Student-t quantiles, key = degrees of freedom
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}
_Z975 = 1.959964


def t975(df: int) -> float:
    """Two-sided 95% t quantile (conservative: rounds df down to a tabulated value)."""
    if df < 1:
        raise ValueError("need at least 2 samples")
    return _T975[max(k for k in _T975 if k <= df)]


def mean_ci_t(values: Iterable[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% t-interval. Half-width is NaN for one value."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("no values")
    mean = sum(xs) / n
    if n == 1:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, t975(n - 1) * math.sqrt(var / n)


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid counts")
    p, z = successes / n, _Z975
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_results(results_dir: Path | str, pattern: str = "*.json") -> list[dict[str, Any]]:
    """Load and validate every results JSON in ``results_dir``."""
    from jmem.evaluate import validate_result

    out = []
    for path in sorted(Path(results_dir).glob(pattern)):
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_result(result)
        result["_path"] = path.name
        out.append(result)
    return out


def _pert_key(p: dict[str, Any]) -> str:
    rest = ",".join(f"{k}={p[k]}" for k in sorted(p) if k != "name")
    return p["name"] + (f"({rest})" if rest else "")


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (env, predictor, perturbation, context, offset) and aggregate over seeds.

    Success values are returned in percent. A seed is a (train_seed, eval_seed)
    pair; duplicates raise, because averaging the same run twice shrinks the CI.
    """
    groups: dict[tuple, list[tuple[tuple, float, int, int]]] = defaultdict(list)
    for r in results:
        for off, frac in r["success"].items():
            key = (r["env"], r["predictor"], _pert_key(r["perturbation"]),
                   r.get("context", ""), int(off))
            n = r["n_episodes"]
            groups[key].append(((r["train_seed"], r["eval_seed"]), frac, round(frac * n), n))
    rows = []
    for key in sorted(groups):
        items = groups[key]
        seeds = [s for s, *_ in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds for {key}: {seeds}")
        mean, half = mean_ci_t([100 * f for _, f, _, _ in items])
        k, n = sum(i[2] for i in items), sum(i[3] for i in items)
        lo, hi = wilson_interval(k, n)
        rows.append({
            "env": key[0], "predictor": key[1], "perturbation": key[2], "context": key[3],
            "offset": key[4], "n_seeds": len(items), "n_episodes_total": n,
            "per_seed": [round(100 * f, 4) for _, f, _, _ in items],
            "mean": mean, "ci_seed": half, "ci_wilson": [100 * lo, 100 * hi],
        })
    return rows


def fmt_mean_ci(mean: float, half: float) -> str:
    """``87.3 ± 4.1`` (one decimal); half-width ``n/a`` for a single seed."""
    return f"{mean:.1f} ± {half:.1f}" if not math.isnan(half) else f"{mean:.1f} ± n/a"


def crosstab(plan: Iterable[bool], replay: Iterable[bool]) -> dict[str, int]:
    """Per-episode cross-tabulation of planner success against dataset-replay success.

    Both sequences must refer to the same episodes in the same order.
    ``plan_fail_replay_fail`` counts planner failures on episodes where even the
    dataset's own actions do not reach the goal (not attributable to the model).
    """
    plan, replay = [bool(x) for x in plan], [bool(x) for x in replay]
    if len(plan) != len(replay):
        raise ValueError("plan and replay cover different numbers of episodes")
    out = {"n": len(plan), "plan_ok_replay_ok": 0, "plan_ok_replay_fail": 0,
           "plan_fail_replay_ok": 0, "plan_fail_replay_fail": 0}
    for p, r in zip(plan, replay):
        out[f"plan_{'ok' if p else 'fail'}_replay_{'ok' if r else 'fail'}"] += 1
    return out


def replace_block(text: str, name: str, body: str) -> str:
    """Replace the content between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->``.

    Used to inject generated tables into README / reports so numbers are never
    typed by hand. Raises if the markers are missing or duplicated.
    """
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise ValueError(f"expected exactly one ordered marker pair for block {name!r}")
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body.strip()}\n{end}{tail}"


def reproduction_table(
    rows: list[dict[str, Any]], references: dict[str, dict[str, Any]], tolerance: float = 5.0
) -> str:
    """Markdown table comparing our reproduction with reference values.

    ``references`` maps env id -> ``{"value": float | None, "source": str}``. A row
    is PASS when ``|mean - value| <= tolerance``; without a reference value the
    verdict is ``TODO(reference)`` (never guessed).
    """
    lines = [
        "| Env | Seeds | Episodes | Success % (mean ± 95% t-CI over seeds) | Wilson 95% (pooled) "
        f"| Per seed | Reference | Δ | Within ±{tolerance:g} |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ref = references.get(r["env"], {})
        value = ref.get("value")
        if value is None:
            ref_s, delta_s, verdict = "TODO(reference)", "—", "TODO(reference)"
        else:
            delta = r["mean"] - float(value)
            ref_s = f"{float(value):.1f} ({ref.get('source', '?')})"
            delta_s = f"{delta:+.1f}"
            verdict = "PASS" if abs(delta) <= tolerance else "FAIL"
        lines.append(
            f"| {r['env']} | {r['n_seeds']} | {r['n_episodes_total']} | "
            f"{fmt_mean_ci(r['mean'], r['ci_seed'])} | "
            f"[{r['ci_wilson'][0]:.1f}, {r['ci_wilson'][1]:.1f}] | "
            f"{', '.join(f'{v:.0f}' for v in r['per_seed'])} | {ref_s} | {delta_s} | {verdict} |"
        )
    return "\n".join(lines)
