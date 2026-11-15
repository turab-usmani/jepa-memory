from datetime import datetime, timedelta, timezone

import pytest

from jmem import ledger

T0 = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_active():
    ledger._active = None
    ledger._active_path = None
    yield
    ledger._active = None
    ledger._active_path = None


def _run(path, run_id, hours, est=1.0, gpu="L4", t0=T0):
    ledger.start(run_id, est_hours=est, notebook="nb", gpu=gpu, path=path, now=t0)
    return ledger.stop(now=t0 + timedelta(hours=hours))


def test_start_stop_writes_one_complete_row(tmp_path):
    path = tmp_path / "ledger.csv"
    row = _run(path, "a", hours=1.5, est=2.0)
    rows = ledger.read(path)
    assert len(rows) == 1
    assert list(rows[0].keys()) == ledger.FIELDS
    assert rows[0] == row
    assert float(rows[0]["hours"]) == pytest.approx(1.5)
    assert float(rows[0]["est_hours"]) == pytest.approx(2.0)
    assert rows[0]["gpu"] == "L4" and rows[0]["notebook"] == "nb"


def test_rows_are_appended_not_overwritten(tmp_path):
    path = tmp_path / "ledger.csv"
    _run(path, "a", 1.0)
    _run(path, "b", 0.25)
    _run(path, "c", 2.0)
    rows = ledger.read(path)
    assert [r["run_id"] for r in rows] == ["a", "b", "c"]
    assert [float(r["hours"]) for r in rows] == pytest.approx([1.0, 0.25, 2.0])
    assert path.read_text().count("run_id") == 1  # single header


def test_summary_arithmetic_and_gpu_conversion(tmp_path):
    path = tmp_path / "ledger.csv"
    _run(path, "l4", 10.0, gpu="L4")
    _run(path, "a100", 4.0, gpu="A100")  # 4 A100-h == 10 L4-h
    _run(path, "cpu", 100.0, gpu="cpu")  # free
    _run(path, "t4", 2.0, gpu="T4")  # unknown GPU charged 1:1
    s = ledger.summary(path, verbose=False)
    assert s.used == pytest.approx(22.0)
    assert s.projected == pytest.approx(22.0)
    assert s.remaining == pytest.approx(128.0)
    assert s.fraction_used == pytest.approx(22.0 / 150.0)
    assert not s.over_budget


def test_budget_equivalence_150_l4_is_60_a100():
    assert ledger.l4_equivalent(60.0, "A100") == pytest.approx(ledger.BUDGET_L4_HOURS)


def test_summary_warns_when_projection_exceeds_budget(tmp_path, capsys):
    path = tmp_path / "ledger.csv"
    _run(path, "big", 140.0)
    ok = ledger.summary(path, planned_hours=10.0, verbose=False)
    assert not ok.over_budget  # exactly on budget is allowed
    s = ledger.summary(path, planned_hours=10.5)
    assert s.over_budget and s.projected == pytest.approx(150.5)
    assert "WARNING" in capsys.readouterr().out


def test_open_row_is_charged_at_estimate(tmp_path):
    path = tmp_path / "ledger.csv"
    _run(path, "done", 1.0)
    ledger.start("crashed", est_hours=3.0, gpu="L4", path=path, now=T0)
    ledger._active = None  # simulate a dead Colab session: stop() never called
    ledger._active_path = None
    s = ledger.summary(path, verbose=False)
    assert s.used == pytest.approx(1.0)
    assert s.open_est == pytest.approx(3.0)
    assert s.projected == pytest.approx(4.0)
    assert s.open_runs == ("crashed",)
    ledger.close_open("crashed", hours=2.0, path=path)
    s = ledger.summary(path, verbose=False)
    assert s.used == pytest.approx(3.0) and s.open_runs == ()


def test_misuse_raises(tmp_path):
    path = tmp_path / "ledger.csv"
    with pytest.raises(RuntimeError):
        ledger.stop()
    ledger.start("a", est_hours=1.0, gpu="L4", path=path, now=T0)
    with pytest.raises(RuntimeError):
        ledger.start("b", est_hours=1.0, gpu="L4", path=path, now=T0)
    with pytest.raises(ValueError):
        ledger.stop(now=T0 - timedelta(hours=1))
    ledger.stop(now=T0 + timedelta(hours=1))
    with pytest.raises(ValueError):
        ledger.start("c", est_hours=-1.0, gpu="L4", path=path)
    with pytest.raises(KeyError):
        ledger.close_open("missing", 1.0, path=path)


def test_empty_ledger(tmp_path):
    s = ledger.summary(tmp_path / "none.csv", verbose=False)
    assert s.used == 0.0 and s.projected == 0.0 and not s.over_budget
