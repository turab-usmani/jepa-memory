"""Filesystem layout for local runs and Google Colab.

Layout (see NOTES.md, "Storage layout"):

* ``root``       persistent project root. Colab: ``/content/drive/MyDrive/jmem``.
                 Local: the repository root.
* ``swm_home``   ``$STABLEWM_HOME``. stable-worldmodel 0.1.1 keeps checkpoints in
                 ``<swm_home>/checkpoints`` (wm/utils.py:77). Persistent.
* ``data_home``  dataset cache root; swm reads ``<data_home>/datasets/<name>.h5``
                 (data/formats/hdf5.py:54-55). On Colab this is the *local* disk
                 (``/content/jmem_data``) because the released datasets are
                 3-46 GB compressed and do not fit a free Drive.
* ``results``    small results JSON files (mirrored into the repo's ``results/``).
* ``checkpoints`` our own training checkpoints (resumable runs).
* ``ledger_csv`` compute ledger.

Every location can be overridden with an environment variable
(``JMEM_ROOT``, ``JMEM_DATA_HOME``, ``STABLEWM_HOME``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

COLAB_DRIVE_MOUNT = Path("/content/drive")
COLAB_ROOT = COLAB_DRIVE_MOUNT / "MyDrive" / "jmem"
COLAB_DATA_HOME = Path("/content/jmem_data")


def repo_root() -> Path:
    """Return the repository root (the directory holding ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


@dataclass(frozen=True)
class Paths:
    """Resolved project locations. All values are absolute paths."""

    root: Path
    swm_home: Path
    data_home: Path
    results: Path
    checkpoints: Path
    ledger_csv: Path
    locks: Path
    lewm_dir: Path

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable view (used in resolved configs)."""
        return {k: str(v) for k, v in asdict(self).items()}


def get_paths(colab: bool | None = None, env: dict[str, str] | None = None) -> Paths:
    """Resolve project paths without touching the filesystem.

    Args:
        colab: Force Colab (True) or local (False) layout. ``None`` auto-detects.
        env: Environment mapping to read overrides from (defaults to ``os.environ``).
    """
    env = os.environ if env is None else env
    colab = in_colab() if colab is None else colab

    if "JMEM_ROOT" in env:
        root = Path(env["JMEM_ROOT"])
    elif colab:
        root = COLAB_ROOT
    else:
        root = repo_root()

    if "STABLEWM_HOME" in env:
        swm_home = Path(env["STABLEWM_HOME"])
    else:
        swm_home = root / ("swm_home" if colab else ".swm_home")

    if "JMEM_DATA_HOME" in env:
        data_home = Path(env["JMEM_DATA_HOME"])
    else:
        data_home = COLAB_DATA_HOME if colab else root / ".jmem_data"

    return Paths(
        root=root,
        swm_home=swm_home,
        data_home=data_home,
        results=root / "results",
        checkpoints=root / "checkpoints",
        ledger_csv=root / "ledger" / "compute_ledger.csv",
        locks=root / "locks",
        lewm_dir=repo_root() / "third_party" / "le-wm",
    )


def mount_drive() -> bool:
    """Mount Google Drive on Colab. Returns True if a mount is present afterwards."""
    if not in_colab():
        return False
    if not (COLAB_DRIVE_MOUNT / "MyDrive").exists():
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(str(COLAB_DRIVE_MOUNT))
    return (COLAB_DRIVE_MOUNT / "MyDrive").exists()


def setup(mount: bool = True) -> Paths:
    """Mount Drive (Colab), create directories and export ``STABLEWM_HOME``.

    Call this once at the top of every notebook / entrypoint.
    """
    if mount and in_colab() and "JMEM_ROOT" not in os.environ:
        if not mount_drive():
            raise RuntimeError("Google Drive is not mounted; cannot persist results.")
    paths = get_paths()
    for d in (
        paths.root,
        paths.swm_home,
        paths.data_home / "datasets",
        paths.results,
        paths.checkpoints,
        paths.ledger_csv.parent,
        paths.locks,
    ):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["STABLEWM_HOME"] = str(paths.swm_home)
    # le-wm/train.py:55 and swm scripts/train/lewm.py:99 read this for datasets.
    os.environ["LOCAL_DATASET_DIR"] = str(paths.data_home)
    return paths
