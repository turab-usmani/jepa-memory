import pytest

from jmem._official_train_runner import summarise_step_times
from jmem.train import clips_per_epoch, estimate_run_hours, official_train_command


def test_summarise_step_times_skips_warmup():
    # 10 intervals: two slow warm-up steps (5 s) then eight 1 s steps
    stamps = [0.0, 5.0, 10.0] + [10.0 + i for i in range(1, 9)]
    s = summarise_step_times(stamps)
    assert s["n_warmup_skipped"] == 2 and s["n_intervals"] == 8
    assert s["median_s"] == pytest.approx(1.0) and s["mean_s"] == pytest.approx(1.0)


def test_summarise_step_times_edge_cases():
    assert summarise_step_times([])["median_s"] is None
    assert summarise_step_times([1.0])["median_s"] is None
    s = summarise_step_times([0.0, 2.0])  # a single interval is kept, not skipped
    assert s["n_intervals"] == 1 and s["median_s"] == pytest.approx(2.0)


def test_clips_per_epoch_matches_swm_indexing():
    # span = 4 * 5 = 20; an episode of length L yields L - 19 clips, shorter ones none
    assert clips_per_epoch([20], 4, 5, 1.0, 1) == 1
    assert clips_per_epoch([19, 10], 4, 5, 1.0, 1) == 0
    # 4 episodes of 40 steps -> 84 clips, 90 % -> 76, batch 4 -> 19 (drop_last);
    # this is the value Lightning reported in the local dry run of le-wm/train.py
    assert clips_per_epoch([40] * 4, 4, 5, 0.9, 4) == 19


def test_clips_per_epoch_on_released_tworoom_totals():
    # 10,000 episodes / 920,809 steps (notebook 00); every episode is >= 31 steps,
    # so clips = 920,809 - 10,000 * 19, independent of how lengths are distributed
    lengths = [92] * 9999 + [920809 - 92 * 9999]
    assert sum(lengths) == 920809 and min(lengths) >= 31
    assert clips_per_epoch(lengths, 4, 5, 0.9, 128) == 5138


def test_estimate_run_hours():
    rec = {"step_time": {"median_s": 0.36}}
    est = estimate_run_hours(rec, epochs=100, steps_per_epoch=5000)
    assert est["total_steps"] == 500_000
    assert est["hours_per_epoch"] == pytest.approx(0.5)
    assert est["hours_total"] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        estimate_run_hours({"step_time": {"median_s": None}}, 1, 1)
    with pytest.raises(ValueError):
        estimate_run_hours(rec, 0, 10)


def test_official_train_command(tmp_path):
    cmd = official_train_command("tworoom", "probe", tmp_path / "h", ["+trainer.max_steps=300"])
    assert cmd[1:3] == ["-m", "jmem._official_train_runner"]
    assert "data=tworoom" in cmd and "subdir=probe" in cmd and "output_model_name=probe" in cmd
    assert cmd[-1] == "+trainer.max_steps=300"
