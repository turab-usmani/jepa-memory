# CHECKPOINT 0 — after Phase 0 (setup and smoke test)

Report date: 2026-11-16. Status: **criteria met; waiting for the human's decision to start Phase 1.**

## Results table

Source: the "NB00 SUMMARY" block pasted by the human from the Colab run of
`notebooks/00_setup_and_smoke.ipynb` at commit `1bc651a` (NVIDIA L4). The
underlying JSON records are on Drive under `jmem/results/smoke/` and are **not
yet copied into the repo's `results/`**.

| Env | Protocol | Episodes | Success (%) | Protocol mismatches | Eval wall-clock (s) | Record |
|---|---|---|---|---|---|---|
| TwoRoom | official `eval.py`, released ckpt | 2 | 100.0 | none | 4.7 | `results/smoke/official_tworoom_n2_s42.json` |
| PushT | official `eval.py`, released ckpt | 2 | 100.0 | none | 3.1 | `results/smoke/official_pusht_n2_s42.json` |

These are **pipeline checks, not success estimates**: two episodes carry no
statistical information and must not be quoted as reproduction results.

## Compute used versus budget

0.105 / 150 L4-hours (0.07 %). Notebook 00 was estimated at 0.5 L4-h and took
0.1046 h (ledger row `nb00-smoke`), including both dataset downloads.

## Verdict per criterion

| Gate 0 criterion | Verdict | Evidence |
|---|---|---|
| Notebook 00 passes on a fresh Colab runtime | PASS | all cells ran; `tests_ok: True`; validation asserts passed; summary pasted back |
| `NOTES.md` explains LeWM with file:line refs | PASS | `NOTES.md` §1–2 |

Facts confirmed by the run:
- The planner sees one real frame: `info_pixels_shape = [2, 1, 224, 224, 3]` on both envs.
- Runtime plan / solver settings equal `configs/eval/protocol.yaml` (no mismatches).
- Model sizes match the local measurement: 18,034,478 total, 10,791,360 predictor.
- Datasets: TwoRoom 10,000 episodes / 920,809 steps (length 31–101, column `ep_idx`);
  PushT 18,685 episodes / 2,336,736 steps (length 49–246, column `episode_idx`).
- Colab stack: Python 3.13.16, torch 2.14.1+cu130, transformers 4.57.6 (downgrade worked).

## Risks

1. Smoke JSONs live only on Drive; results used in any table must be copied into
   the repo's `results/` (notebook 01 will do this explicitly).
2. `wm-all` (Reacher, Cube) is still untested; those datasets are 23.8 GB and 46.2 GB.
3. Scientific: with D9 decided, P1/P2 use our own planning loop. It must first
   reproduce the official number in 1-frame mode. Q2–Q4 (`NOTES.md` §4) are open.
4. Per-episode evaluation is cheap (about 2 s per episode here), so evaluation
   will not be the budget bottleneck; training cost is still unmeasured.

## Recommended next step

Start Phase 1, step 5.1 (reproduction), after the human approves and answers Q2–Q4:
released checkpoints, official protocol, 50 episodes × 3 eval seeds on TwoRoom and
PushT. Extrapolating from the smoke run (assumes linear scaling in episodes,
unverified): well under 0.5 L4-h in total. The from-scratch TwoRoom training run
that measures L4-hours per run would follow, with its own estimate first.
