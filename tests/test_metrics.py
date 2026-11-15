import copy

import jsonschema
import pytest

from jmem.evaluate import (
    load_protocol,
    official_record_to_result,
    success_rate,
    validate_result,
    write_result,
)


def make_record(successes, seed=42, env="tworoom"):
    """A synthetic record shaped like the output of run_official_eval."""
    protocol = load_protocol()
    common, spec = protocol["common"], protocol["envs"][env]
    n = len(successes)
    return {
        "env": env, "seed": seed, "git_sha": "abc", "hardware": "L4", "wallclock_s": 12.5,
        "timestamp": "2026-11-16T00:00:00+00:00", "versions": {"torch": "2.14.1"},
        "num_envs": n, "max_episode_steps": common["max_episode_steps"],
        "info_pixels_shape": [n, 1, 224, 224, 3],
        "evaluate_kwargs": {
            "goal_offset": 25, "eval_budget": 50, "episodes_idx": list(range(n)),
            "start_steps": [0] * n, "callables": copy.deepcopy(spec["callables"]),
        },
        "metrics": {"success_rate": 100.0 * sum(successes) / n, "episode_successes": successes},
        "policy": {
            "type": "WorldModelPolicy", "plan_config": dict(common["plan"]),
            "solver": {"type": "stable_worldmodel.solver.cem.CEMSolver",
                       **{k: v for k, v in common["solver"].items() if k != "target"}},
            "model": {"predictor_num_frames": 3, "n_params_predictor": 10791360},
        },
        "hydra_config": {
            "seed": seed, "eval": {"img_size": 224, "dataset_name": spec["dataset_name"]},
            "world": {"env_name": spec["env_name"]},
        },
    }


def test_success_on_hand_made_trajectories():
    assert success_rate([True, False, False, True]) == 0.5
    assert success_rate([False] * 7) == 0.0
    assert success_rate([True] * 3) == 1.0
    assert success_rate([1, 0, 1]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        success_rate([])


def test_official_record_converts_and_validates():
    successes = [True] * 43 + [False] * 7
    res = official_record_to_result(make_record(successes), load_protocol())
    assert res["success"] == {"25": 0.86}
    assert res["n_episodes"] == 50
    assert res["env"] == "swm/TwoRoom-v1"
    assert res["predictor"] == "transformer_h3" and res["params_predictor"] == 10791360
    assert res["train_seed"] is None and res["eval_seed"] == 42
    assert res["perturbation"] == {"name": "none"}
    assert res["context"] == "official_1frame"
    assert res["protocol"]["mismatches"] == []
    assert res["protocol"]["common"]["goal_offset_steps"] == 25
    # not measured by the official script: must be null, never a fake 0.0
    assert res["rollout_mse"] is None and res["probe_r2"] is None
    assert res["latency_ms_median"] is None


def test_mismatches_are_carried_into_the_result():
    rec = make_record([True, False] * 25)
    rec["policy"]["solver"]["n_steps"] = 10
    res = official_record_to_result(rec, load_protocol())
    assert len(res["protocol"]["mismatches"]) == 1
    assert res["protocol"]["mismatches"][0].startswith("solver.n_steps")


def test_schema_rejects_bad_results(tmp_path):
    good = official_record_to_result(make_record([True, False]), load_protocol())
    path = write_result(good, tmp_path)
    assert path.name == "official-tworoom-released-s42.json"

    bad = copy.deepcopy(good)
    bad["success"]["25"] = 86.0  # percent instead of fraction
    with pytest.raises(jsonschema.ValidationError):
        validate_result(bad)
    bad = copy.deepcopy(good)
    del bad["eval_seed"]
    with pytest.raises(jsonschema.ValidationError):
        validate_result(bad)
    bad = copy.deepcopy(good)
    bad["success"] = {}
    with pytest.raises(jsonschema.ValidationError):
        validate_result(bad)
    bad = copy.deepcopy(good)
    bad["success"]["25"] = 1.0  # disagrees with the episode list
    with pytest.raises(ValueError):
        validate_result(bad)
    bad = copy.deepcopy(good)
    bad["n_episodes"] = 3
    with pytest.raises(ValueError):
        validate_result(bad)
