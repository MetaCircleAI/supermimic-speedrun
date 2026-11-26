# SuperMimic Speedrun rules

Train a Unitree G1 whole-body motion-tracking policy from scratch on one NVIDIA
H100 SXM 80 GB HBM3. The final policy qualifies when its full-library
`norm_return` is strictly greater than **0.9363**. Rank qualifying entries by
ascending unrounded wall-clock `speedrun_seconds`. One completed run suffices.

## Task and resources

Use Uni-SuperMimic revision `d78a5e2bc26ee26c3a94ae10418b7fb9018836ee`
and the reference task semantics, including the robot, simulator dynamics,
observations, action interface, control rate, reward, resets and termination.
Use the processed `bones_seed_compact.pt` with SHA256
`5d815fd8039c4e4938a6c049935b8d712f81f483116ce69518c8a8298290017a`.
Set both `algo.seed=1` and `env.seed=1` before constructing their respective
components. Record every additional RNG seed and how it derives from seed 1.

Policy and world-model architecture, optimizer, schedule, training sampling,
training environment count, precision and kernels are open. Ordinary host CPU,
RAM and storage are allowed. All run-specific simulation, preprocessing and
learning must use one host and its one designated GPU. Initialize
learned components from scratch. External training data, pretrained components,
teacher outputs from earlier runs, and saved experience or learned states from
other runs are ineligible.

At evaluation the policy receives only the upstream policy observations and
reference command, with the upstream lookahead. It may use its own observation
history within the episode. Reset episode state on reset and freeze learned
parameters. Clip identifiers, simulator internals, evaluation outcomes and
per-clip action tables are outside the policy interface. Architecture changes
must preserve this boundary.

## Timing and checkpoint selection

Use a monotonic wall clock. The reference run starts immediately before the
first simulator step of replay prefill. For other recipes, start before the
first run-specific computation that supplies training with experience, targets,
features or parameter updates, whichever occurs first. Include simulator work,
all updates, in-loop compilation, graph capture, and checkpoint writes. Stop
after queued accelerator work and the final checkpoint writer have finished.

Imports, loading the fixed inputs, ordinary environment construction, and
final evaluation are outside the clock. A compiler cache from the same code
and hardware may contain compiled kernels. Disclose its use. Any reusable
artifact containing learned, rollout-derived or candidate-specific data must
be produced inside the run. Development and hyperparameter search are allowed.

Before starting a record attempt, save its configuration, code identity,
seeds and fixed training budget. Run continuously from scratch to that budget.
For new submissions, include the launch-time declaration in the training log
so reviewers can match its budget and hashes to the final checkpoint.
Only the final checkpoint is eligible. Intermediate recovery snapshots are
allowed. Resumed runs, post-hoc checkpoint selection and evaluation-driven
early stopping are ineligible.

## Scoring

Use the reference `infra_v3_terminal_bin_fix` evaluator (`train/eval_one.py` with
`train/runtime/boundary_fix.py`) with deterministic policy rollout, evaluation seed 0,
16,384 parallel environments and all 129,487 motions. Evaluate the submitted final weights once. Every motion in the fixed
library contributes exactly once, including failed rollouts under the upstream
termination rule. Compute each clip's normalized return with the upstream
denominator and take the unweighted arithmetic mean across clips. The frozen
evaluator's full-precision `norm_return` decides qualification.
`norm_return_framew` and failure count are supplementary.

## Submission and review

Supply the pre-run configuration and budget, training and evaluation commands,
source revision and changes, dependencies, hardware and cache settings, training
log, `training_complete.json`, final checkpoint and SHA256, evaluation summary,
and per-clip results, in a new `records/<date>_<name>/` folder. Include the resolved
training seeds and the checkpoint hash consumed by evaluation. Machine-local paths may be
made portable. Publish per-clip results with clip names replaced by their SHA256
(`tools/hash_clip_names.py`); the BONES-SEED license keeps clip names confidential.

Reviewers check the timer boundaries, run identity, clip coverage and score.
For a new entry, maintainers review the policy adapter and evaluate the submitted
weights using their own pinned task, motion library and scorer. This independent
check uses the same final checkpoint. Maintainers may request a witnessed timed
run if the wall time or resource use needs confirmation. `tools/verify_records.py`
checks the published records.
