# CHECKPOINT 0 — after Phase 0 (setup and smoke test)

Date: 2026-11-16. Status: **WAITING FOR THE COLAB RUN** (not passed yet).

## Results table

No experimental results exist yet.

| Env | Protocol | Success (%) | Source JSON |
|---|---|---|---|
| TwoRoom | official, released ckpt, 2 episodes (smoke) | TODO(result) | `results/smoke/official_tworoom_n2_s42.json` |
| PushT | official, released ckpt, 2 episodes (smoke) | TODO(result) | `results/smoke/official_pusht_n2_s42.json` |

(Two-episode smoke numbers prove the pipeline runs; they are not success estimates.)

## Compute used versus budget

0.0 / 150 L4-hours (ledger is empty; all work so far was local CPU).
Notebook 00 estimate: ≤ 0.5 L4-h.

## Verdict per criterion

| Gate 0 criterion | Verdict | Evidence |
|---|---|---|
| `NOTES.md` explains LeWM with file:line refs | PASS | `NOTES.md` §1–2 |
| Notebook 00 passes on a fresh Colab runtime | **PENDING** | not executed on Colab; needs the human |

What was verified locally (Windows, Python 3.13, CPU):
- `pytest -m "not slow"`: 49 passed with the LeWM stack installed, 35 passed + 5
  skipped without it; < 1 min.
- Released TwoRoom weights load with the pinned stack and `get_cost` runs.
- The unmodified `le-wm/eval.py` runs end-to-end through our capture runner
  (`policy=random`, synthetic 4-episode TwoRoom dataset).
- All notebook code cells parse.

What was NOT verified: the CUDA policy path, the real datasets (archive layout,
column names), Colab's preinstalled packages, PushT, wall-clock times.

## Risks

1. **Install on Colab.** `transformers` must be downgraded below 5 (released
   weights do not load otherwise), which drags `huggingface_hub<1` and
   `datasets<5.1`. pip may ask for a runtime restart or report conflicts with
   unrelated preinstalled packages.
2. **Dataset layout.** `tworoom.tar.zst` is assumed to contain exactly one `.h5`;
   the code raises a clear error listing the archive contents otherwise.
3. **Episode-index column.** `eval.py` needs `ep_idx` or `episode_idx` in the HDF5.
4. **Disk.** PushT is 13.1 GB compressed; uncompressed size unknown.
5. **Scientific (see NOTES §2.4–2.5, §4):** the official planner sees one real
   frame and makes two decisions per episode. The POMDP benchmark and the
   baseline definition depend on Q1–Q4, which need a decision before Phase 1.

## Recommended next step

1. Human: push the repo, run `notebooks/00_setup_and_smoke.ipynb` on an L4, paste
   back the "NB00 SUMMARY" block (or the first failing cell's output).
2. Human: answer Q1–Q4 in `NOTES.md` §4.
3. Agent: fix whatever notebook 00 reveals; after Gate 0 passes, start Phase 1
   (5.1 reproduction: 50 episodes × 3 seeds on TwoRoom and PushT).
