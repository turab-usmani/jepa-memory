# jepa-memory

Memory for JEPA world models under partial observability, built on
[LeWorldModel](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

Status: **Phase 0** (setup + smoke test). No experimental results yet.

## Layout

```
configs/          base.yaml (LeWM training defaults), eval/protocol.yaml (THE eval protocol)
src/jmem/         package: paths, ledger, utils, datasets, evaluate (+ official-eval runner)
tests/            unit tests (pytest -m "not slow")
notebooks/        Colab notebooks (thin; logic lives in src/jmem)
third_party/le-wm git submodule, pinned (see NOTES.md)
results/          small results JSON (committed)
reports/          checkpoint-gate reports
NOTES.md          how LeWM actually works, discrepancies, decisions, open questions
```

Modules for later phases (`pomdp`, `predictors`, `model`, `train`, `plan`, `probes`,
`aggregate`) are added when they have tested content.

## Local development (CPU)

```bash
git clone --recurse-submodules <repo-url> && cd jepa-memory
pip install -e ".[dev]"          # light: torch, numpy, pyyaml, jsonschema, pytest
pytest -m "not slow" -q
pip install -e ".[wm,dev]"       # optional: pinned LeWM stack, enables the swm-dependent tests
```

## Colab

1. Push this repo to a remote you can clone from Colab.
2. Open `notebooks/00_setup_and_smoke.ipynb`, select an L4 (or A100) runtime.
3. Set `REPO_URL` in the config cell, then *Run all*.

Results, checkpoints, the compute ledger and pip lock files go to
`/content/drive/MyDrive/jmem/`. Datasets go to the Colab local disk
(`/content/jmem_data`) because they are 3–46 GB (NOTES.md, D3).

## Results

| Env | Protocol | Success (%) | Source |
|---|---|---|---|
| TwoRoom | official config, released checkpoint | TODO(result) | `results/` |
| PushT | official config, released checkpoint | TODO(result) | `results/` |

Every number in this table must be generated from a results JSON by
`jmem.aggregate` (Phase 1); nothing is typed by hand.

## Compute budget

150 L4-hours (or 60 A100-hours). Tracked in
`<drive>/jmem/ledger/compute_ledger.csv` via `jmem.ledger`.
