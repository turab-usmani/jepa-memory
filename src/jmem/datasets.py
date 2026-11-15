"""Locate / download the released LeWM datasets and checkpoints.

Sources (verified 2026-11-16 via the Hugging Face API, see NOTES.md):

* datasets  ``https://huggingface.co/datasets/quentinll/lewm-*`` (zstd archives)
* weights   ``https://huggingface.co/quentinll/lewm-*`` (``weights.pt`` + ``config.json``)

swm 0.1.1 reads HDF5 datasets from ``<cache_dir>/datasets/<name>.h5``
(``stable_worldmodel/data/formats/hdf5.py:54-55``) and resolves a
``<user>/<repo>`` checkpoint name by downloading it into
``$STABLEWM_HOME/checkpoints/models--<user>--<repo>/``
(``stable_worldmodel/wm/utils.py:141-160``).
"""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    """One released dataset archive."""

    hf_repo: str
    hf_file: str
    # name passed to swm (``eval.dataset_name`` in le-wm/config/eval/<env>.yaml);
    # the file lives at <data_home>/datasets/<name>.h5
    name: str
    compressed_gb: float


# Only the two Phase-0 environments are registered. Reacher (23.8 GB) and Cube
# (46.2 GB) are added in Phase 1 once their archive layout has been inspected.
DATASETS: dict[str, DatasetSpec] = {
    "tworoom": DatasetSpec("quentinll/lewm-tworooms", "tworoom.tar.zst", "tworoom", 3.43),
    "pusht": DatasetSpec(
        "quentinll/lewm-pusht", "pusht_expert_train.h5.zst", "pusht_expert_train", 13.14
    ),
}

# ``policy=`` value for le-wm/eval.py (resolved by swm.wm.utils.load_pretrained).
CHECKPOINTS: dict[str, str] = {
    "tworoom": "quentinll/lewm-tworooms",
    "pusht": "quentinll/lewm-pusht",
    "reacher": "quentinll/lewm-reacher",
    "cube": "quentinll/lewm-cube",
}


def dataset_path(env: str, data_home: Path | str) -> Path:
    """Where swm expects the HDF5 file for ``env``."""
    return Path(data_home) / "datasets" / f"{DATASETS[env].name}.h5"


def decompress_zst(src: Path | str, dst: Path | str, chunk: int = 1 << 22) -> Path:
    """Stream-decompress a ``.zst`` file to ``dst`` (written atomically)."""
    import zstandard

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with src.open("rb") as fin, tmp.open("wb") as fout:
        zstandard.ZstdDecompressor().copy_stream(fin, fout, read_size=chunk, write_size=chunk)
    tmp.replace(dst)
    return dst


def extract_tar_zst(src: Path | str, dst_dir: Path | str) -> list[Path]:
    """Extract a ``.tar.zst`` archive into ``dst_dir``; return the extracted files."""
    import zstandard

    src, dst_dir = Path(src), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin:
        with zstandard.ZstdDecompressor().stream_reader(fin) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                # 'data' filter: refuses absolute paths, links outside dst, devices.
                tar.extractall(dst_dir, filter="data")
    return sorted(p for p in dst_dir.rglob("*") if p.is_file())


def install_archive(archive: Path | str, target: Path | str) -> Path:
    """Unpack a downloaded archive so that the HDF5 file ends up at ``target``.

    Handles the two layouts on the Hub: a bare ``<name>.h5.zst`` and a
    ``<name>.tar.zst`` containing exactly one ``*.h5`` file.
    """
    archive, target = Path(archive), Path(target)
    if archive.name.endswith(".tar.zst"):
        work = target.parent / f".extract_{target.stem}"
        if work.exists():
            shutil.rmtree(work)
        files = extract_tar_zst(archive, work)
        h5 = [p for p in files if p.suffix in (".h5", ".hdf5")]
        if len(h5) != 1:
            raise RuntimeError(
                f"expected exactly one HDF5 file in {archive.name}, found: "
                f"{[str(p.relative_to(work)) for p in files]}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(h5[0]), str(target))
        shutil.rmtree(work)
    elif archive.name.endswith(".zst"):
        decompress_zst(archive, target)
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")
    return target


def ensure_dataset(env: str, data_home: Path | str, keep_archive: bool = False) -> Path:
    """Return the local HDF5 path for ``env``, downloading it from the Hub if absent."""
    target = dataset_path(env, data_home)
    if target.exists():
        print(f"[data] {env}: found {target}")
        return target
    spec = DATASETS[env]
    from huggingface_hub import hf_hub_download

    archives = Path(data_home) / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    print(f"[data] {env}: downloading {spec.hf_repo}/{spec.hf_file} (~{spec.compressed_gb} GB)")
    archive = Path(
        hf_hub_download(spec.hf_repo, spec.hf_file, repo_type="dataset", local_dir=archives)
    )
    print(f"[data] {env}: unpacking to {target}")
    install_archive(archive, target)
    if not keep_archive:
        archive.unlink(missing_ok=True)
    return target


def describe_h5(path: Path | str) -> dict[str, Any]:
    """Columns, shapes, dtypes and episode count of an swm HDF5 dataset."""
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers the compression filters)
    except ImportError:
        pass
    out: dict[str, Any] = {"path": str(path), "columns": {}}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            ds = f[k]
            out["columns"][k] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
        if "ep_len" in f:
            ep_len = f["ep_len"][:]
            out["n_episodes"] = int(len(ep_len))
            out["n_steps"] = int(ep_len.sum())
            out["ep_len_min_max"] = [int(ep_len.min()), int(ep_len.max())]
    return out


def ensure_checkpoint(env: str) -> Path:
    """Download (if needed) the released LeWM weights for ``env``; return their folder.

    Uses the same resolver as ``le-wm/eval.py:88`` so the files land exactly
    where the official script will look for them.
    """
    from stable_worldmodel.data import get_cache_dir
    from stable_worldmodel.wm.utils import _resolve

    ckpt_root = get_cache_dir(None, sub_folder="checkpoints")
    weights, _config = _resolve(CHECKPOINTS[env], ckpt_root)
    return Path(weights).parent
