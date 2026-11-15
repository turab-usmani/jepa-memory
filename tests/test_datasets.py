import io
import tarfile

import numpy as np
import pytest

from jmem import datasets

zstandard = pytest.importorskip("zstandard")
h5py = pytest.importorskip("h5py")


def _make_h5(path, n_eps=3, ep_len=5):
    with h5py.File(path, "w") as f:
        f["pixels"] = np.zeros((n_eps * ep_len, 4, 4, 3), dtype=np.uint8)
        f["action"] = np.arange(n_eps * ep_len * 2, dtype=np.float32).reshape(-1, 2)
        f["ep_len"] = np.full(n_eps, ep_len)
        f["ep_offset"] = np.arange(n_eps) * ep_len


def _zst(src, dst):
    dst.write_bytes(zstandard.ZstdCompressor().compress(src.read_bytes()))


def test_dataset_path_matches_swm_layout(tmp_path):
    assert datasets.dataset_path("tworoom", tmp_path) == tmp_path / "datasets" / "tworoom.h5"
    assert (
        datasets.dataset_path("pusht", tmp_path)
        == tmp_path / "datasets" / "pusht_expert_train.h5"
    )


def test_install_bare_zst(tmp_path):
    raw = tmp_path / "pusht_expert_train.h5"
    _make_h5(raw)
    archive = tmp_path / "pusht_expert_train.h5.zst"
    _zst(raw, archive)
    target = tmp_path / "out" / "datasets" / "pusht_expert_train.h5"
    datasets.install_archive(archive, target)
    assert target.read_bytes() == raw.read_bytes()
    assert not target.with_name(target.name + ".part").exists()


@pytest.mark.parametrize("inner", ["tworoom.h5", "some/nested/dir/tworoom.h5"])
def test_install_tar_zst(tmp_path, inner):
    raw = tmp_path / "raw.h5"
    _make_h5(raw)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(raw, arcname=inner)
    archive = tmp_path / "tworoom.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    target = tmp_path / "out" / "datasets" / "tworoom.h5"
    datasets.install_archive(archive, target)
    assert target.read_bytes() == raw.read_bytes()
    assert [p.name for p in target.parent.iterdir()] == ["tworoom.h5"]  # work dir removed


def test_install_tar_with_two_h5_is_rejected(tmp_path):
    raw = tmp_path / "raw.h5"
    _make_h5(raw)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(raw, arcname="a.h5")
        tar.add(raw, arcname="b.h5")
    archive = tmp_path / "x.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    with pytest.raises(RuntimeError, match="exactly one HDF5"):
        datasets.install_archive(archive, tmp_path / "out" / "x.h5")


def test_ensure_dataset_uses_existing_file_without_network(tmp_path):
    target = datasets.dataset_path("tworoom", tmp_path)
    target.parent.mkdir(parents=True)
    _make_h5(target)
    assert datasets.ensure_dataset("tworoom", tmp_path) == target


def test_describe_h5(tmp_path):
    path = tmp_path / "d.h5"
    _make_h5(path, n_eps=3, ep_len=5)
    info = datasets.describe_h5(path)
    assert info["n_episodes"] == 3 and info["n_steps"] == 15
    assert info["ep_len_min_max"] == [5, 5]
    assert info["columns"]["pixels"]["shape"] == [15, 4, 4, 3]
    assert info["columns"]["action"]["dtype"] == "float32"


def test_registry_is_consistent_with_protocol():
    from jmem.evaluate import load_protocol

    protocol = load_protocol()
    for env, spec in datasets.DATASETS.items():
        assert protocol["envs"][env]["dataset_name"] == spec.name
    for env, repo in datasets.CHECKPOINTS.items():
        assert protocol["envs"][env]["checkpoint"] == repo
