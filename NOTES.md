# NOTES — running log

Newest entries at the bottom of each section. File:line references are to the
pinned sources:

| Source | Pin | Where |
|---|---|---|
| `le-wm` | commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (2026-05-22, "fix data config") | git submodule `third_party/le-wm` |
| `stable-worldmodel` (swm) | PyPI **0.1.1** = tag `0.1.1` = commit `15a5538d492ae524c64cb18cc56a2d70611e877e` (2026-06-06) | pip; a read-only clone may sit in `third_party/stable-worldmodel` (git-ignored) |
| `stable-pretraining` (spt) | PyPI **0.1.8** | pip |

swm `main` (`21446f1`, 2026-10-02) is 51 files / +4964 −851 lines ahead of 0.1.1 inside
the package. We pin the PyPI release, not `main`. The 0.1.1 wheel equals the tag
except for two stray directories (`envs/humanoid_bench`, `planning`) that only
exist in the wheel.

---

## 1. How LeWM actually works (with file:line refs)

Paths starting `le-wm/` are in the submodule; paths starting `swm/` are
`stable_worldmodel/` at tag 0.1.1.

**Two copies of the model.** `le-wm/jepa.py` (`JEPA`) + `le-wm/module.py` is the
paper repo. swm ships its own copy, `swm/wm/lewm/lewm.py` (`LeWM`) +
`swm/wm/lewm/module.py`. The **released checkpoints instantiate the swm copy**:
their `config.json` has `_target_: stable_worldmodel.wm.lewm.LeWM` and
`stable_worldmodel.wm.lewm.module.Predictor`. `le-wm/train.py` trains the
`jepa.JEPA` copy (`le-wm/config/train/model/lewm.yaml:1`). The two differ only in
inference plumbing (see 2.6).

**Model** (`le-wm/config/train/model/lewm.yaml`, `le-wm/jepa.py`):
- Encoder: HF ViT-tiny, patch 14, 224 px, random init (`model/lewm.yaml:3-9`);
  the CLS token is the frame embedding (`jepa.py:37-38`), followed by a
  projector MLP 192→2048→192 with BatchNorm1d (`model/lewm.yaml:29-36`, `module.py:217-241`).
- Action encoder `Embedder`: Conv1d(k=1) then MLP to 192 (`module.py:189-214`).
  Its input width is `frameskip × action_dim`, set in code (`train.py:68`).
- Predictor `ARPredictor`: 6 `ConditionalBlock`s, 16 heads × 64, MLP 2048,
  dropout 0.1, learned positional embedding of length `num_frames = history_size`
  (`module.py:244-285`, `model/lewm.yaml:11-22`). Causal self-attention
  (`module.py:83`, `is_causal=True`). Action conditioning is **AdaLN-zero**: the
  action embedding produces shift/scale/gate for both sub-layers, zero-initialised
  (`module.py:88-111`).
- `pred_proj`: second BatchNorm MLP applied to predictor outputs (`jepa.py:53`).

**Training step** (`le-wm/train.py:17-45`): a clip of `history_size + num_preds = 4`
frames (frameskip 5) is encoded; `pred = predict(emb[:, :3], act_emb[:, :3])`,
target `emb[:, 1:]` — teacher-forced next-embedding prediction at all 3 positions,
**no stop-gradient, no EMA**. Loss = MSE + 0.09 × SIGReg(emb) (`train.py:39-41`).
SIGReg is an Epps–Pulley statistic on 1024 random 1-D projections, 17 knots
(`module.py:10-36`), applied over the batch dimension per time step
(`train.py:40`, input transposed to (T, B, D)).
Optimiser AdamW lr 5e-5, wd 1e-3, bf16, grad-clip 1.0, batch 128, 100 epochs,
90/10 random clip split, seed 3072 (`config/train/lewm.yaml:10-43`).

**Data** (`swm/data/dataset.py:36-72`, `swm/data/formats/hdf5.py:111-132`):
a sample is `num_steps` observations `frameskip` apart; see 2.3 for the action
convention. Transforms: ImageNet normalisation + resize (`le-wm/utils.py:6-10`),
z-score for every non-pixel column (`train.py:62-66`).

**Planning / evaluation** (`le-wm/eval.py`, `swm/world/world.py`, `swm/policy.py`,
`swm/solver/cem.py`):
1. `eval.py:123-134` samples `num_eval` dataset rows whose episode has at least
   `goal_offset_steps` steps left (rng seeded with `cfg.seed`).
2. `World._evaluate_from_dataset` (`world.py:492-564`) resets one env per row,
   forces the simulator into the row's state and the goal state via `callables`
   (`world.py:515-521`, `601-614`), and uses the dataset frame at
   `start + goal_offset` as the goal image (`world.py:567-598`). The goal is
   re-written into `infos` after every step (`world.py:532`, `542`).
3. `WorldModelPolicy.get_action` (`policy.py:349-439`) replans only when its
   action buffer is empty; the buffer holds `receding_horizon × action_block = 25`
   env actions (`policy.py:341-343`). With `eval_budget = 50` that is **2 planner
   calls per episode** (t = 0 and t = 25).
4. `CEMSolver.solve` (`cem.py:118-283`): 300 samples × 30 iterations, top-30,
   in z-scored action space, first sample forced to the mean (`cem.py:207`).
5. Cost (`lewm.py:126-151`, `110-124`): encode the goal image, roll the predictor
   out 5 steps (`lewm.py:58-108`), squared distance between the **last** predicted
   embedding and the goal embedding.
6. Success = `terminated` at any step within the budget (`world.py:543`).
   TwoRoom: agent within 16 px of the goal (`swm/envs/two_room/env.py:273`).
   PushT: position error < 20 and angle error < π/9 (`swm/envs/pusht/env.py:347-355`).

**Checkpoints.** `eval.py:88` calls `swm.wm.utils.load_pretrained(cfg.policy)`,
which accepts a HF repo id and downloads `weights.pt` + `config.json` into
`$STABLEWM_HOME/checkpoints/models--<user>--<repo>/` (`swm/wm/utils.py:47-160`).
Released: `quentinll/lewm-{tworooms,pusht,reacher,cube}` (72 MB each).

---

## 2. Where the code disagrees with the brief (Section 1.1 and elsewhere)

### 2.1 Eval protocol — brief confirmed
`config/eval/{tworoom,pusht,reacher,cube}.yaml`: `eval_budget: 50`,
`goal_offset_steps: 25`, `num_eval: 50`, seed 42. `max_episode_steps` is set to
`2 × eval_budget = 100` in code (`eval.py:57`). Captured in
`configs/eval/protocol.yaml`, enforced by `tests/test_protocol.py`.

### 2.2 Config-vs-paper values — brief confirmed (code values)
history size **3** (`config/train/lewm.yaml:15`), SIGReg weight **0.09** (`:40`),
CEM iterations **30** (`config/eval/solver/cem.yaml:6`), epochs **100** (`:18`).

### 2.3 The four undocumented conventions — all found in code
1. **Dense action gathering.** Observations are strided by `frameskip`, actions
   are not (`swm/data/formats/hdf5.py:121-122`); the action block is reshaped to
   `(num_steps, frameskip × action_dim)` (`swm/data/dataset.py:70-71`). One
   predictor step therefore consumes 5 raw actions (10-D for a 2-D action space).
2. **Action-encoder width** = `frameskip × action_dim`, set at `train.py:68`
   (`???` in the YAML).
3. **ImageNet normalisation**: `le-wm/utils.py:7-8` (train), `eval.py:22` (eval).
4. **Action z-scoring with NaN rows dropped**: `le-wm/utils.py:25-32` (train),
   `eval.py:75-78` (eval).

   Extra details not in the brief:
   - remaining NaN actions are zeroed **after** scaling (`train.py:25`), i.e. they
     become the mean action;
   - train uses `torch.std` (ddof = 1), eval uses sklearn `StandardScaler`
     (ddof = 0): different by a factor √((N−1)/N), negligible but real;
   - `train.py:62-66` also z-scores `proprio` / `state`, which the model never reads.

### 2.4 NEW — the planner sees exactly ONE real frame (claim-relevant)
`PlanConfig.history_len` (`swm/policy.py:31`) is declared and documented but
**never read** anywhere in swm 0.1.1. `EnvPool` always stacks infos as
`(num_envs, 1, ...)` (`swm/world/env_pool.py:162-182`) and nothing accumulates
past frames. So `LeWM.rollout` starts from `H = 1` encoded frame
(`swm/wm/lewm/lewm.py:69`) and fills its 3-slot window with its **own
predictions** (`lewm.py:94-100`). Consequences:
- "History size 3" is a training-time property. At decision time the official
  LeWM is memoryless with respect to real observations.
- With 2 planner calls per episode (2.5) the policy looks at **two frames per
  episode** in total.
- Enforced by `test_swm_never_reads_history_len` and
  `test_swm_infos_have_a_single_time_step`; checked at runtime through
  `info_pixels_shape` in the eval record.

This reshapes RQ1/RQ2 — see "Open questions" Q1.

### 2.5 NEW — horizon arithmetic
`horizon 5 × action_block 5 = 25` env steps = `goal_offset_steps`. The official
task is "reach a state exactly one planning horizon away, with two attempts".
success@offset 50/75 (brief 5.3) puts the goal 2–3 horizons away while the cost
only scores the last imagined step; expect a drop for reasons unrelated to memory.

### 2.6 `jepa.JEPA` vs `swm.wm.lewm.LeWM`
- `JEPA.rollout` defaults `history_size=3` and `get_cost` never passes it
  (`le-wm/jepa.py:61`, `149`); `LeWM.rollout` uses `predictor.num_frames`
  (`swm/wm/lewm/lewm.py:65-66`). Identical for released models; they diverge for
  our `transformer_h{1,8}` variants → we build on the swm class.
- `LeWM` caches `emb` / `goal_emb` across CEM iterations (`lewm.py:77`, `132`);
  `JEPA` re-encodes each iteration. Same numbers, different latency → latency
  must be measured on one class only.
- `LeWM.encode` casts pixels to the encoder dtype (`lewm.py:29`); `JEPA` uses `.float()`.

### 2.7 Storage layout differs from the brief and from the le-wm README
- swm 0.1.1 reads datasets from `<cache>/datasets/<name>.h5`
  (`swm/data/formats/hdf5.py:54-55`) and checkpoints from `<cache>/checkpoints/`
  (`swm/wm/utils.py:77`), not from `$STABLEWM_HOME` directly as the le-wm README says.
- The README's `_object.ckpt` + `AutoCostModel` path is stale; `eval.py:88` uses
  `load_pretrained`. No conversion step is needed for the HF weights.
- Released datasets (HF `datasets/quentinll/lewm-*`, sizes compressed):
  TwoRoom 3.43 GB (`tworoom.tar.zst`), PushT 13.14 GB (`pusht_expert_train.h5.zst`),
  Reacher 23.75 GB, Cube 46.18 GB. **They do not fit a free 15 GB Drive**, and
  HDF5 over the Drive FUSE mount is slow. Decision D3 below.

### 2.8 Smaller mismatches
- `config/train/data/pusht.yaml:4` trains from `pusht_expert_train.lance`, but
  only an `.h5` is released and `eval.py:42` reads HDF5. Training PushT needs
  `data.dataset.name=pusht_expert_train.h5` (or a one-off conversion).
- `eval.py:89` hard-codes `model.to("cuda")`: the official eval cannot run on CPU.
- le-wm README installs `stable-worldmodel[train,env]`; `HDF5Dataset` needs
  `h5py` + `hdf5plugin` from the `format` extra.
- LR schedule: `le-wm/train.py:91` passes no warm-up / max steps to
  `LinearWarmupCosineAnnealingLR`; swm's `scripts/train/lewm.py:150-160` sets
  1 % warm-up and `max_steps`. Which one produced the released weights is unknown.
- swm's own TwoRoom eval config sets state from `pos_agent` / `goal_pos_agent`;
  le-wm uses `proprio` / `goal_proprio` (`config/eval/tworoom.yaml:34-42`). We
  follow le-wm; notebook 00 prints the dataset columns so this can be confirmed.
- **transformers must be < 5.** With transformers 5.23 the HF ViT parameters are
  renamed (`encoder.layers.N.attention.q_proj` instead of
  `encoder.layer.N.attention.attention.query`) and `load_pretrained` fails on the
  released weights. Verified locally: fails on 5.23.0, loads on 4.57.6 (which in
  turn needs `huggingface_hub<1` and `datasets<5.1`). Guarded by `tests/test_compat.py`.
- **CEM `batch_size` must stay 1.** `LeWM.criterion` (`swm/wm/lewm/lewm.py:115`)
  expands `goal_emb` of shape (B, 1, D) to (B, S, T, D), which raises for B > 1
  (reproduced locally with B = 2). Envs are therefore planned one at a time.
- Parameter counts measured on `quentinll/lewm-tworooms` (CPU, local): encoder
  5,501,376; projector 792,768; predictor **10,791,360**; pred_proj 792,768;
  action encoder 156,206; total **18,034,478** (the brief/paper say "about 15M").
- HDF5 files written by swm 0.1.1's own `World.collect` have no `ep_idx` /
  `episode_idx` column, which `eval.py:68` requires. The released files must
  carry one (unverified until notebook 00 prints the columns).
- `eval.py:29-37` computes episode lengths with one boolean mask per episode
  (O(episodes × rows)); may take minutes on PushT. To be timed in notebook 00.

---

## 3. Decisions

- **D1 (2026-11-16)** Pin swm to PyPI 0.1.1, not `main`. Reason: reproducible
  install, and it is what `pip install stable-worldmodel` gives.
- **D2** Official evaluation = `third_party/le-wm/eval.py`, run unmodified through
  `jmem._official_eval_runner`, which only wraps `swm.World.evaluate` to record
  its arguments, planner settings and metrics. This wrapper is also the planned
  injection point for POMDP observation wrappers (Phase 1).
- **D3** Deviation from brief 4.3: `STABLEWM_HOME` is on Drive (checkpoints,
  72 MB each) but **datasets live on the Colab local disk**
  (`/content/jmem_data/datasets`, passed as Hydra `cache_dir=` and
  `LOCAL_DATASET_DIR`) and are re-downloaded per session (reason: 2.7).
  Override with `JMEM_DATA_HOME` if you have Drive space.
- **D4** `torch` is range-pinned (`>=2.4,<3`) so Colab keeps its CUDA build;
  everything LeWM-specific is `==`-pinned. Exact resolved versions are stored in
  every record (`utils.env_versions`) and notebook 00 writes a `pip freeze` lock
  to `<drive>/jmem/locks/`.
- **D5** Two install extras: `wm` (TwoRoom + PushT; no `gymnasium[all]`, no jax)
  and `wm-all` (adds swm's `env` extra for Reacher / Cube). Phase 0 uses `wm`.
- **D6** Ledger charges A100 hours at 2.5 L4-h (150/60), CPU at 0, unknown GPUs 1:1.
- **D7** Extra module `src/jmem/datasets.py` (not in the brief's layout) holds
  download/unpack logic so notebooks stay thin. Layout files for later phases
  (`pomdp.py`, `predictors.py`, …) are created when they have real content.
- **D8** Goal image is never perturbed (brief 5.2) — to be implemented in Phase 1;
  recorded here as required by the brief.

- **D9 (2026-11-16, decided by the human; resolves Q1)** At plan time **every
  predictor variant receives the same stream of real frames**: `transformer_h{H}`
  keeps a FIFO of the last H real latents, recurrent variants update their state
  on each real frame. The official 1-frame protocol is kept only as a separate
  reproduction row (released checkpoint, `le-wm/eval.py` unchanged). Consequence:
  P1/P2 numbers come from our own planning loop (`jmem.plan`), not from swm's
  `WorldModelPolicy`, so that loop must first be shown to reproduce the official
  number when restricted to 1 frame (regression test + notebook 01 check).

- **D10 (2026-11-16, agent default, not yet confirmed — Q3)** Evaluation seeds
  are `[42, 43, 44]`. The seed picks the evaluation episodes and seeds CEM, so
  the three seeds are three different 50-episode sets from the same checkpoint.
  They measure evaluation variance, **not** training variance (one released
  checkpoint per env).
- **D11** Two intervals are always reported and labelled (`jmem.aggregate`):
  a 95% Student-t interval across seeds, and a 95% Wilson interval on the pooled
  episodes. With 3 seeds the t quantile is 4.303, so the seed interval is wide.
- **D12** Reference values for the ±5 check live in `configs/eval/references.yaml`.
  Only TwoRoom (87, from the brief) is filled in; PushT / Reacher / Cube stay
  `TODO(reference)` until the human supplies the LeWM paper's table values.

### Phase 1 findings about the official training script (2026-11-16)

Found by running `le-wm/train.py` unchanged through `jmem._official_train_runner`
on a 4-episode synthetic TwoRoom dataset (CPU, 12 optimisation steps):
- **Cost pre-estimate (to be replaced by the notebook-01 probe).** TwoRoom has
  920,809 − 10,000 × 19 = 730,809 clips; 90 % / batch 128 = **5,138 steps per
  epoch**, so the official 100 epochs are **513,800 steps**. One run costs
  142.7 × (seconds per step) L4-hours: 0.1 s/step → 14 h, 0.3 s/step → 43 h.
  The step time on an L4 is unknown until the probe runs, but a 100-epoch run is
  very unlikely to be "a few hours", and 8–10 perturbed runs plus the P2 grid
  cannot all be 100-epoch runs inside 150 L4-h.
- **No auto-resume as shipped.** `train.py:132-138` resumes only if
  `<run_dir>/<name>_weights.ckpt` exists, and nothing in the dry run wrote that
  file (only `weights_epoch_N.pt` state dicts via `SaveCkptCallback`). A long
  run therefore needs our own resumable checkpointing (brief 6.5).
- **One `.pt` per epoch.** `load_pretrained(<run>)` raises "Ambiguous checkpoint"
  when a folder holds several `.pt` files (`swm/wm/utils.py:132-136`); evaluate
  from-scratch runs with `policy=<run>/weights_epoch_<N>.pt`.
- From-scratch checkpoints carry `_target_: jepa.JEPA`, so loading them needs
  `third_party/le-wm` on `sys.path` (the eval runner already does this) and uses
  the le-wm class, not swm's `LeWM` (see 2.6 for the differences).
- `stable_pretraining.Manager` uses Unix-only signals: training cannot run on
  Windows. The local dry run used a throw-away signal shim.

### Notebook 01 results (2026-11-16, L4, commit cac811e)

Source: the "NB01 SUMMARY" block pasted by the human. The results JSON files are
in `reproduction_results.zip` on Drive and are **not yet committed** to
`results/reproduction/`; until they are, README keeps `TODO(result)`.

- Reproduction, released checkpoints, official protocol, 50 episodes × seeds
  42/43/44, no protocol mismatches:
  - TwoRoom 85.3 ± 2.9 (t-CI over seeds), Wilson [78.8, 90.1], per seed 86/84/86
    (43, 42, 43 of 50). Reference 87 → Δ −1.7 → **within ±5**.
  - PushT 89.3 ± 5.7, Wilson [83.4, 93.3], per seed 92/88/88 (46, 44, 44 of 50).
    No verified reference value yet (`TODO(reference)`).
- Training throughput probe (official `train.py`, 300 steps, batch 128, bf16,
  6 workers, 12 CPU cores): **median 0.601 s/step** (240 intervals, max 0.625).
  With 5,138 steps/epoch: 0.86 L4-h per epoch, **85.8 L4-h for the official 100
  epochs, 8.6 L4-h for 10 epochs** (lower bounds: no validation, no checkpoint
  writes). Not yet known whether the step is GPU-bound or dataloader-bound.
- Compute used so far: 0.165 / 150 L4-h.

**Budget consequence.** One official from-scratch run is 57 % of the whole
budget. The brief's plan (one from-scratch run + 8–10 perturbed LeWM runs in P1 +
a P2 grid of 4 envs × 3–4 conditions × 5–6 predictors × 3 seeds) is not
feasible end-to-end at any epoch count close to the official one. Needs a
decision — Q6 below.

## 4. Open questions (need the human's decision)

- **Q6 — how do we train within 150 L4-h?** (from the probe above)
  (a) *Frozen released encoder, train predictors on cached latents.* Encode each
      dataset once with the released encoder (inference only), then train every
      predictor variant on 192-d latents: minutes per run instead of days, so the
      full grid with 3 seeds fits. Keeps "identical encoder across variants"
      exactly. Cost: not end-to-end; the encoder never adapts to perturbed
      frames; claims must be scoped to "predictor memory on top of a fixed LeWM
      encoder".
  (b) *End-to-end only, heavily cut.* E.g. ≤ 2 epochs per run (1.7 L4-h) and a
      much smaller grid. Faithful to LeWM training but under-trained, few runs,
      weak statistics.
  (c) *Hybrid (recommended).* (a) for the full grid, plus a small number of
      end-to-end confirmation runs (10-epoch, ~8.6 L4-h each) on the most
      informative condition to check that the frozen-encoder conclusion survives.
  The from-scratch TwoRoom reproduction run (brief 5.1) at the official 100
  epochs should not be launched under any option; a 10-epoch version is the most
  we can afford, and only if you want it.

- **Q1 — RESOLVED by D9 (option b).** Original question: what does the baseline see at plan time? (from 2.4) Under the official
  protocol no predictor receives observation history, so a perturbed current
  frame is all the planner has, and a recurrent predictor could not help either
  unless the policy feeds it every frame. Options:
  (a) *official* context (1 frame) for `transformer_h*`, per-step state updates
      for recurrent variants — maximally faithful to LeWM but confounds "memory
      architecture" with "being fed more frames";
  (b) give **every** variant the same stream of real frames (FIFO of the last H
      for transformers, state update for recurrent), and report the official
      1-frame protocol as an extra row;
  (c) both, as a 2×N table.
  Recommendation: (b), with the official row kept for the reproduction.
- **Q2 — decision frequency.** Two decisions per episode (2.5) means `flicker`
  can only bite at the single replan frame (frame 0 is never blanked), and
  `delay` shifts two frames. Should POMDP evaluation replan more often
  (e.g. `receding_horizon: 1`, 10 decisions per episode) so that perturbations
  and memory have something to act on? This changes the protocol, so the clean
  numbers must then be reported under both settings.
- **Q3 — evaluation seeds.** Brief asks for 3 eval seeds; the official config has
  one (42, which also picks the episodes). Proposed `[42, 43, 44]`.
- **Q4 — success@offset 50/75.** Keep `eval_budget = 50` and `horizon = 5` fixed
  (pure "official protocol for everything else"), or scale budget with offset?
- **Q5 — repo hosting.** Notebooks clone `REPO_URL`; a GitHub (or other) remote is
  needed. Not created by the agent.

## 5. Log

- 2026-11-16 Notebook 00 ran on Colab (L4, commit 1bc651a, 0.1046 h): all cells
  passed, no protocol mismatches. Confirms 2.4 at runtime (pixels shape
  `[2, 1, 224, 224, 3]`). Resolves two "unverified" items in 2.8: the released
  files do carry an episode-index column (`ep_idx` in TwoRoom, `episode_idx` in
  PushT), and TwoRoom has both `proprio` and `pos_agent`, so le-wm's callables
  work as written. The episode-length loop in `eval.py:29-37` is not a problem
  (total wall-clock 33 s TwoRoom, 52 s PushT for 2 episodes, model load included).
  Colab runs Python 3.13.16 / torch 2.14.1+cu130; the transformers downgrade to
  4.57.6 installed cleanly. Gate 0 criteria met; see `reports/CHECKPOINT_0.md`.
- 2026-11-16 Local verification (Windows, Python 3.13, CPU, torch 2.15): fast
  tests pass with and without the LeWM stack; released TwoRoom weights load and
  `get_cost` runs from a single frame; the unmodified `eval.py` runs end-to-end
  through `jmem._official_eval_runner` with `policy=random` on a 4-episode
  synthetic TwoRoom dataset. NOT verified locally: anything needing CUDA (the
  LeWM policy path), the real datasets, Colab's package set, PushT.
- 2026-11-16 Phase 0: repos pinned, sources read, sections 1–4 written. Package
  skeleton (`paths`, `ledger`, `utils`, `datasets`, `evaluate`, runner), configs
  (`base.yaml`, `eval/protocol.yaml`), tests, notebook 00. Notebook 00 has **not**
  been executed on Colab yet (Gate 0 pending).
