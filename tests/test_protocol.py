"""configs/eval/protocol.yaml and configs/base.yaml must equal what the official code reads.

Three layers:
1. the le-wm YAML configs (always checked; needs the third_party/le-wm submodule);
2. the stable-worldmodel defaults / behaviour (checked when swm is installed);
3. a runtime record produced by jmem.evaluate.run_official_eval (logic checked here
   on a synthetic record; the real comparison runs in the notebooks).
"""

import copy
import inspect
import subprocess
from pathlib import Path

import pytest

from jmem.evaluate import check_protocol, load_protocol, official_eval_command
from jmem.paths import repo_root
from jmem.utils import load_yaml

LEWM = repo_root() / "third_party" / "le-wm"
ENVS = ["tworoom", "pusht", "reacher", "cube"]


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


@pytest.fixture(scope="module")
def lewm_dir():
    assert (LEWM / "eval.py").exists(), (
        "third_party/le-wm is missing; run `git submodule update --init`"
    )
    return LEWM


def _official_eval_cfg(lewm_dir: Path, env: str) -> dict:
    cfg = load_yaml(lewm_dir / "config" / "eval" / f"{env}.yaml")
    # Hydra defaults list: every env config composes `solver: cem`
    assert {"solver": "cem"} in cfg["defaults"]
    cfg["solver"] = load_yaml(lewm_dir / "config" / "eval" / "solver" / "cem.yaml")
    return cfg


def test_pinned_lewm_commit(protocol, lewm_dir):
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=lewm_dir, capture_output=True, text=True
    )
    if out.returncode != 0:
        pytest.skip("git not available")
    assert out.stdout.strip() == protocol["source"]["le_wm_commit"]


@pytest.mark.parametrize("env", ENVS)
def test_protocol_matches_official_eval_config(protocol, lewm_dir, env):
    cfg = _official_eval_cfg(lewm_dir, env)
    common, spec = protocol["common"], protocol["envs"][env]

    assert cfg["eval"]["num_eval"] == common["num_eval"]
    assert cfg["eval"]["goal_offset_steps"] == common["goal_offset_steps"]
    assert cfg["eval"]["eval_budget"] == common["eval_budget"]
    assert cfg["eval"]["img_size"] == common["img_size"]
    assert cfg["seed"] == common["seed"]

    for key in ("horizon", "receding_horizon", "action_block"):
        assert cfg["plan_config"][key] == common["plan"][key]
    # history_len / warm_start are NOT set by the official config -> swm defaults apply
    assert set(cfg["plan_config"]) == {"horizon", "receding_horizon", "action_block"}

    assert cfg["solver"]["_target_"] == common["solver"]["target"]
    for key in ("num_samples", "n_steps", "topk", "var_scale", "batch_size"):
        assert cfg["solver"][key] == common["solver"][key]

    assert cfg["world"]["env_name"] == spec["env_name"]
    assert cfg["eval"]["dataset_name"] == spec["dataset_name"]
    assert cfg["dataset"]["keys_to_cache"] == spec["keys_to_cache"]
    assert cfg["eval"]["callables"] == spec["callables"]
    extra = {
        k: v
        for k, v in cfg["world"].items()
        if k not in ("env_name", "num_envs", "max_episode_steps")
    }
    assert extra == spec["world_kwargs"]
    assert cfg["world"]["num_envs"] == "${eval.num_eval}"


def test_protocol_matches_eval_script(protocol, lewm_dir):
    src = (lewm_dir / "eval.py").read_text(encoding="utf-8")
    common = protocol["common"]
    assert "cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget" in src
    assert common["max_episode_steps"] == 2 * common["eval_budget"]
    assert "image_shape=(224, 224)" in src
    assert common["world_image_shape"] == [224, 224]
    assert "g = np.random.default_rng(cfg.seed)" in src
    assert "max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1" in src
    assert "transforms.Normalize(**spt.data.dataset_stats.ImageNet)" in src
    assert "goal_offset=cfg.eval.goal_offset_steps" in src


def test_derived_quantities(protocol):
    common = protocol["common"]
    plan = common["plan"]
    env_steps_per_plan = plan["receding_horizon"] * plan["action_block"]
    assert common["planner_calls_per_episode"] == common["eval_budget"] / env_steps_per_plan
    # the planning horizon reaches exactly the goal offset
    assert plan["horizon"] * plan["action_block"] == common["goal_offset_steps"]
    assert protocol["eval_seeds"][0] == common["seed"]
    assert len(set(protocol["eval_seeds"])) == 3


def test_base_matches_lewm_train_config(lewm_dir):
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    train = load_yaml(lewm_dir / "config" / "train" / "lewm.yaml")
    model = load_yaml(lewm_dir / "config" / "train" / "model" / "lewm.yaml")
    t = base["train"]
    assert t["seed"] == train["seed"]
    assert t["train_split"] == train["train_split"]
    assert t["img_size"] == train["img_size"]
    assert t["embed_dim"] == train["embed_dim"]
    assert t["history_size"] == train["history_size"] == 3
    assert t["num_preds"] == train["num_preds"]
    assert t["max_epochs"] == train["trainer"]["max_epochs"] == 100
    assert t["precision"] == train["trainer"]["precision"]
    assert t["gradient_clip_val"] == train["trainer"]["gradient_clip_val"]
    assert t["batch_size"] == train["loader"]["batch_size"]
    assert t["num_workers"] == train["num_workers"]
    assert t["optimizer"]["type"] == train["optimizer"]["type"]
    assert float(t["optimizer"]["lr"]) == float(train["optimizer"]["lr"])
    assert float(t["optimizer"]["weight_decay"]) == float(train["optimizer"]["weight_decay"])
    assert t["sigreg"]["weight"] == train["loss"]["sigreg"]["weight"] == 0.09
    assert t["sigreg"]["knots"] == train["loss"]["sigreg"]["kwargs"]["knots"]
    assert t["sigreg"]["num_proj"] == train["loss"]["sigreg"]["kwargs"]["num_proj"]

    for key in ("depth", "heads", "dim_head", "mlp_dim", "dropout", "emb_dropout"):
        assert base["model"]["predictor"][key] == model["predictor"][key]
    assert base["model"]["encoder"]["size"] == model["encoder"]["size"]
    assert base["model"]["encoder"]["patch_size"] == model["encoder"]["patch_size"]
    assert model["action_encoder"]["input_dim"] == "???"  # set in code (train.py:68)

    for name in ("pusht", "tworoom", "dmc", "ogb"):
        data = load_yaml(lewm_dir / "config" / "train" / "data" / f"{name}.yaml")
        assert data["dataset"]["frameskip"] == base["data"]["frameskip"]
    assert base["data"]["num_steps"] == t["num_preds"] + t["history_size"]

    train_src = (lewm_dir / "train.py").read_text(encoding="utf-8")
    assert (
        "cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * "
        'dataset.get_dim("action")' in train_src
    )
    assert 'batch["action"] = torch.nan_to_num(batch["action"], 0.0)' in train_src
    utils_src = (lewm_dir / "utils.py").read_text(encoding="utf-8")
    assert "data = data[~torch.isnan(data).any(dim=1)]" in utils_src
    assert "imagenet_stats = dt.dataset_stats.ImageNet" in utils_src


# --------------------------------------------------------------------------- #
# Layer 2: stable-worldmodel itself (skipped when it is not installed)
# --------------------------------------------------------------------------- #


def test_swm_defaults_match_protocol(protocol):
    swm = pytest.importorskip("stable_worldmodel")
    from importlib import metadata

    assert metadata.version("stable-worldmodel") == protocol["source"]["stable_worldmodel_version"]

    plan = protocol["common"]["plan"]
    cfg = swm.PlanConfig(
        horizon=plan["horizon"],
        receding_horizon=plan["receding_horizon"],
        action_block=plan["action_block"],
    )
    assert cfg.history_len == plan["history_len"]
    assert cfg.warm_start == plan["warm_start"]
    assert cfg.plan_len == protocol["common"]["goal_offset_steps"]

    sig = inspect.signature(swm.solver.CEMSolver.__init__).parameters
    for key in ("num_samples", "n_steps", "topk", "var_scale", "batch_size"):
        assert sig[key].default == protocol["common"]["solver"][key]


def test_swm_never_reads_history_len():
    """PlanConfig.history_len is declared but unused: the planner sees one frame."""
    swm = pytest.importorskip("stable_worldmodel")
    pkg = Path(swm.__file__).parent
    users = [
        p.relative_to(pkg).as_posix()
        for p in pkg.rglob("*.py")
        if "history_len" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    # policy.py only declares/documents it; data/buffer.py is an unrelated ReplayBuffer arg
    assert sorted(users) == ["data/buffer.py", "policy.py"]
    policy_src = (pkg / "policy.py").read_text(encoding="utf-8")
    assert "cfg.history_len" not in policy_src


def test_swm_infos_have_a_single_time_step(protocol):
    pytest.importorskip("stable_worldmodel")
    import numpy as np
    from stable_worldmodel.world.env_pool import _stack_fresh

    infos = [{"pixels": np.zeros((8, 8, 3), dtype=np.uint8)} for _ in range(4)]
    stacked = _stack_fresh(infos)
    assert stacked["pixels"].shape == (4, 1, 8, 8, 3)
    assert stacked["pixels"].shape[1] == protocol["common"]["context_frames_at_plan_time"]


# --------------------------------------------------------------------------- #
# Layer 3: runtime-record comparison logic
# --------------------------------------------------------------------------- #


def _record(protocol: dict, env: str) -> dict:
    common, spec = protocol["common"], protocol["envs"][env]
    n = common["num_eval"]
    return {
        "evaluate_kwargs": {
            "goal_offset": common["goal_offset_steps"],
            "eval_budget": common["eval_budget"],
            "episodes_idx": list(range(n)),
            "start_steps": [0] * n,
            "callables": copy.deepcopy(spec["callables"]),
        },
        "num_envs": n,
        "max_episode_steps": common["max_episode_steps"],
        "info_pixels_shape": [n, 1, 224, 224, 3],
        "policy": {
            "plan_config": dict(common["plan"]),
            "solver": {
                "type": "stable_worldmodel.solver.cem.CEMSolver",
                **{k: v for k, v in common["solver"].items() if k != "target"},
            },
        },
        "hydra_config": {
            "seed": common["seed"],
            "eval": {"img_size": common["img_size"], "dataset_name": spec["dataset_name"]},
            "world": {"env_name": spec["env_name"], "max_episode_steps": "???"},
        },
    }


@pytest.mark.parametrize("env", ENVS)
def test_check_protocol_accepts_a_conforming_run(protocol, env):
    assert check_protocol(_record(protocol, env), protocol, env) == []


def test_check_protocol_flags_the_paper_appendix_protocol(protocol):
    """The appendix-F.1 settings (150 steps, goal 100 ahead, 10 CEM iters) must be caught."""
    rec = _record(protocol, "tworoom")
    rec["evaluate_kwargs"]["goal_offset"] = 100
    rec["evaluate_kwargs"]["eval_budget"] = 150
    rec["policy"]["solver"]["n_steps"] = 10
    bad = check_protocol(rec, protocol, "tworoom")
    assert len(bad) == 3
    assert any(m.startswith("goal_offset_steps") for m in bad)
    assert any(m.startswith("eval_budget") for m in bad)
    assert any(m.startswith("solver.n_steps") for m in bad)


def test_check_protocol_flags_goal_and_history_changes(protocol):
    rec = _record(protocol, "pusht")
    rec["evaluate_kwargs"]["callables"][1]["args"]["goal_state"]["value"] = "state"
    rec["info_pixels_shape"][1] = 3
    bad = check_protocol(rec, protocol, "pusht")
    assert any(m.startswith("callables") for m in bad)
    assert any(m.startswith("context_frames_at_plan_time") for m in bad)


def test_check_protocol_seeds(protocol):
    """Regression (notebook 01): the extra eval seeds are part of the protocol."""
    for seed in protocol["eval_seeds"]:
        rec = _record(protocol, "tworoom")
        rec["seed"] = seed
        rec["hydra_config"]["seed"] = seed
        assert check_protocol(rec, protocol, "tworoom") == []
    rec = _record(protocol, "tworoom")
    rec["seed"] = 7
    rec["hydra_config"]["seed"] = 7  # requested and used, but not a protocol seed
    assert check_protocol(rec, protocol, "tworoom") == [
        f"seed: protocol eval_seeds={protocol['eval_seeds']!r} runtime=7"
    ]
    rec = _record(protocol, "tworoom")
    rec["seed"] = 43  # requested 43 but the script ran with 42
    bad = check_protocol(rec, protocol, "tworoom")
    assert bad == ["seed: protocol=43 runtime=42"]


def test_check_protocol_ignore(protocol):
    rec = _record(protocol, "tworoom")
    rec["num_envs"] = 2
    rec["evaluate_kwargs"]["episodes_idx"] = [0, 1]
    assert len(check_protocol(rec, protocol, "tworoom")) == 2
    assert check_protocol(rec, protocol, "tworoom", ignore=["num_eval"]) == []


def test_official_eval_command(tmp_path):
    cmd = official_eval_command(
        "tworoom", "quentinll/lewm-tworooms", 2, 42, tmp_path / "data", tmp_path / "h",
        extra_overrides=["solver.n_steps=30"],
    )
    assert cmd[1:3] == ["-m", "jmem._official_eval_runner"]
    assert "--config-name=tworoom" in cmd
    assert "policy=quentinll/lewm-tworooms" in cmd
    assert "eval.num_eval=2" in cmd and "seed=42" in cmd
    assert f"cache_dir={(tmp_path / 'data').as_posix()}" in cmd
    assert cmd[-1] == "solver.n_steps=30"
