import json
import math

import pytest

from jmem import aggregate
from jmem.evaluate import load_protocol, official_record_to_result, write_result

from test_metrics import make_record


def test_mean_ci_t_hand_computed():
    mean, half = aggregate.mean_ci_t([80.0, 84.0, 88.0])
    assert mean == pytest.approx(84.0)
    # sd = 4, se = 4 / sqrt(3), t(0.975, df=2) = 4.303
    assert half == pytest.approx(4.303 * 4 / math.sqrt(3), rel=1e-6)
    mean, half = aggregate.mean_ci_t([50.0])
    assert mean == 50.0 and math.isnan(half)
    assert aggregate.mean_ci_t([70.0, 70.0, 70.0]) == (70.0, 0.0)
    with pytest.raises(ValueError):
        aggregate.mean_ci_t([])


def test_t_quantiles():
    assert aggregate.t975(2) == 4.303
    assert aggregate.t975(22) == aggregate.t975(20)  # conservative rounding down
    with pytest.raises(ValueError):
        aggregate.t975(0)


def test_wilson_interval_known_values():
    lo, hi = aggregate.wilson_interval(42, 50)  # 84 %
    assert lo == pytest.approx(0.7149, abs=2e-3) and hi == pytest.approx(0.9166, abs=2e-3)
    lo, hi = aggregate.wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.31
    lo, hi = aggregate.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and hi <= 1.0 and 0.69 < lo < 0.75
    with pytest.raises(ValueError):
        aggregate.wilson_interval(5, 0)


def _write(tmp_path, counts, env="tworoom"):
    protocol = load_protocol()
    for seed, k in counts.items():
        rec = make_record([True] * k + [False] * (50 - k), seed=seed, env=env)
        write_result(official_record_to_result(rec, protocol), tmp_path)


def test_summarize_over_seeds(tmp_path):
    _write(tmp_path, {42: 40, 43: 42, 44: 44})
    _write(tmp_path, {42: 45}, env="pusht")
    rows = aggregate.summarize(aggregate.load_results(tmp_path))
    by_env = {r["env"]: r for r in rows}
    tw = by_env["swm/TwoRoom-v1"]
    assert tw["n_seeds"] == 3 and tw["n_episodes_total"] == 150 and tw["offset"] == 25
    assert tw["per_seed"] == [80.0, 84.0, 88.0]
    assert tw["mean"] == pytest.approx(84.0)
    assert tw["ci_seed"] == pytest.approx(4.303 * 4 / math.sqrt(3), rel=1e-6)
    lo, hi = aggregate.wilson_interval(126, 150)
    assert tw["ci_wilson"] == pytest.approx([100 * lo, 100 * hi])
    pt = by_env["swm/PushT-v1"]
    assert pt["n_seeds"] == 1 and math.isnan(pt["ci_seed"]) and pt["mean"] == pytest.approx(90.0)


def test_duplicate_seed_is_rejected(tmp_path):
    _write(tmp_path, {42: 40})
    res = aggregate.load_results(tmp_path)
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate.summarize(res + res)


def test_load_results_validates(tmp_path):
    _write(tmp_path, {42: 40})
    path = next(tmp_path.glob("*.json"))
    bad = json.loads(path.read_text())
    bad["success"]["25"] = 2.0
    path.write_text(json.dumps(bad))
    with pytest.raises(Exception):
        aggregate.load_results(tmp_path)


def test_crosstab():
    ct = aggregate.crosstab([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert ct == {"n": 5, "plan_ok_replay_ok": 2, "plan_ok_replay_fail": 1,
                  "plan_fail_replay_ok": 1, "plan_fail_replay_fail": 1}
    assert sum(v for k, v in ct.items() if k != "n") == ct["n"]
    with pytest.raises(ValueError):
        aggregate.crosstab([True], [True, False])


def test_replace_block():
    text = "a\n<!-- BEGIN:t -->\nold\n<!-- END:t -->\nb\n"
    out = aggregate.replace_block(text, "t", "new 1\nnew 2\n")
    assert out == "a\n<!-- BEGIN:t -->\nnew 1\nnew 2\n<!-- END:t -->\nb\n"
    assert aggregate.replace_block(out, "t", "new 1\nnew 2") == out  # idempotent
    with pytest.raises(ValueError):
        aggregate.replace_block("no markers", "t", "x")
    with pytest.raises(ValueError):
        aggregate.replace_block(text + text, "t", "x")


def test_reproduction_table_verdicts(tmp_path):
    _write(tmp_path, {42: 40, 43: 42, 44: 44})
    _write(tmp_path, {42: 30, 43: 30, 44: 30}, env="pusht")
    rows = aggregate.summarize(aggregate.load_results(tmp_path))
    refs = {"swm/TwoRoom-v1": {"value": 87.0, "source": "brief"}, "swm/PushT-v1": {"value": None}}
    table = aggregate.reproduction_table(rows, refs)
    tw = next(line for line in table.splitlines() if "TwoRoom" in line)
    assert "84.0 ± 9.9" in tw and "87.0 (brief)" in tw and "-3.0" in tw
    assert tw.rstrip(" |").endswith("PASS")
    pt = next(line for line in table.splitlines() if "PushT" in line)
    assert pt.count("TODO(reference)") == 2 and "PASS" not in pt and "FAIL" not in pt
    refs["swm/TwoRoom-v1"]["value"] = 95.0
    assert "FAIL" in aggregate.reproduction_table(rows, refs)
