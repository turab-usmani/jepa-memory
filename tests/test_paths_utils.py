import json
from pathlib import Path

import pytest
import torch

from jmem import paths, utils


def test_local_paths_are_repo_relative():
    p = paths.get_paths(colab=False, env={})
    root = paths.repo_root()
    assert (root / "pyproject.toml").exists()
    assert p.root == root
    assert p.results == root / "results"
    assert p.swm_home == root / ".swm_home"
    assert p.lewm_dir == root / "third_party" / "le-wm"


def test_colab_paths_follow_the_brief():
    p = paths.get_paths(colab=True, env={})
    assert p.root == Path("/content/drive/MyDrive/jmem")
    assert p.swm_home == Path("/content/drive/MyDrive/jmem/swm_home")
    assert p.results == Path("/content/drive/MyDrive/jmem/results")
    assert p.checkpoints == Path("/content/drive/MyDrive/jmem/checkpoints")
    assert p.ledger_csv.parent == Path("/content/drive/MyDrive/jmem/ledger")
    # datasets are too large for Drive: local disk (documented deviation)
    assert p.data_home == Path("/content/jmem_data")


def test_env_overrides(tmp_path):
    env = {
        "JMEM_ROOT": str(tmp_path / "r"),
        "STABLEWM_HOME": str(tmp_path / "s"),
        "JMEM_DATA_HOME": str(tmp_path / "d"),
    }
    p = paths.get_paths(colab=True, env=env)
    assert (p.root, p.swm_home, p.data_home) == (tmp_path / "r", tmp_path / "s", tmp_path / "d")
    assert all(isinstance(v, str) for v in p.as_dict().values())


def test_setup_creates_dirs_and_exports_env(tmp_path, monkeypatch):
    monkeypatch.setenv("JMEM_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("STABLEWM_HOME", raising=False)
    monkeypatch.setenv("JMEM_DATA_HOME", str(tmp_path / "data"))
    # registered before setup() mutates os.environ so teardown restores the originals
    monkeypatch.setenv("LOCAL_DATASET_DIR", "placeholder")
    p = paths.setup(mount=False)
    assert p.results.is_dir() and p.checkpoints.is_dir() and p.swm_home.is_dir()
    assert (p.data_home / "datasets").is_dir()
    import os

    assert os.environ["STABLEWM_HOME"] == str(p.swm_home)
    assert os.environ["LOCAL_DATASET_DIR"] == str(p.data_home)


def test_seed_everything_is_reproducible():
    utils.seed_everything(123)
    a = torch.rand(4)
    utils.seed_everything(123)
    b = torch.rand(4)
    utils.seed_everything(124)
    c = torch.rand(4)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_count_params():
    lin = torch.nn.Linear(3, 5)  # 15 + 5
    assert utils.count_params(lin) == 20
    lin.bias.requires_grad_(False)
    assert utils.count_params(lin, trainable_only=True) == 15
    shared = torch.nn.ModuleList([lin, lin])  # shared tensors counted once
    assert utils.count_params(shared) == 20


@pytest.mark.parametrize(
    "name,label",
    [
        ("NVIDIA L4", "L4"),
        ("NVIDIA A100-SXM4-40GB", "A100"),
        ("Tesla T4", "T4"),
        ("NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"),
    ],
)
def test_classify_gpu(name, label):
    assert utils.classify_gpu(name) == label


def test_resolve_and_save_config(tmp_path):
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = utils.resolve_config(base, {"a": {"y": 5}, "c": 7})
    assert merged == {"a": {"x": 1, "y": 5}, "b": 3, "c": 7}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}  # inputs untouched
    out = utils.save_resolved_config(merged, tmp_path, name="run")
    payload = json.loads(out.read_text())
    assert payload["config"] == merged
    assert {"git_sha", "versions", "hardware", "timestamp"} <= payload.keys()
