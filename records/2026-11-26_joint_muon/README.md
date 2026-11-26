# 16.08 minutes on one H100 (Muon)

On November 26, 2026, the Joint-then-separate token policy (candidate `return_speed_v1_t002`,
the same policy as the 27.57-minute record) reached 0.9393044114 normalized return on the full
BONES-SEED library after 1,600 iterations. The training timer measured 964.838 seconds from
the first simulator step of the replay prefill through the final checkpoint write. The
evaluation used seed 0 and reported 590 failed clips out of 129,487.
[Record](RECORD.json) and
[evaluation summary](original/eval/001600.summary.json).

## What changed from the 27.57-minute record

The policy architecture, data, simulator, sampling and reward are unchanged. Only the optimizer
recipe differs:

| | 27.57-minute record | This record |
| :--- | :--- | :--- |
| Policy optimizer | Adam | Muon on hidden weight matrices, Adam on the rest |
| World-model optimizer | Adam | Muon on hidden weight matrices, Adam on the rest |
| Learning rate | 8e-4 | 1.2e-3 |
| Iterations | 2,600 | 1,600 |

Muon is Uni-SuperMimic's own `MuonAdam` split (`match_rms_adamw` scaling, momentum 0.95,
Nesterov). In the world model its two hidden 2048×2048 matrices use Muon. In the policy all
attention, SwiGLU and readout matrices use Muon (15 tensors, 1.62 M parameters); the two input
stems, the action head, biases and normalizers stay on Adam (43 k parameters). The split
actually used is recorded in [optimizer_audit.json](original/optimizer_audit.json). Policy
EMA 0.98, 100 world-model-only iterations, BF16, compilation and the WM CUDA graph are as in
the previous record. The critic stays on Adam; one learning rate is shared by all optimizers, so
it also runs at 1.2e-3. To run Muon inside the WM CUDA graph, `hooks.py` switches the WM
optimizer to Uni-SuperMimic's tensor-learning-rate (`capturable`) Muon path. The run reused
compiler caches from earlier runs of the same code on the same hardware.

## Files

- [Policy specification](snapshot/candidate_spec.json) and [training code](snapshot/code/)
  ([Muon integration](snapshot/code/runtime/hooks.py), [recipe overrides](snapshot/code/common.py))
- The run was launched with `snapshot/code/run_variant.py <variant> r1600_env1 <gpu> 1600 --eval full`,
  which trains and then evaluates the final checkpoint. [pipeline.py](pipeline.py) does the same
  for reproductions and also writes `RECORD.json`; this record's
  [RECORD.json](RECORD.json) was assembled afterwards from its training and evaluation outputs.
  [Command with portable paths](command.json).
- [Launch-time declaration](original/launch_request.json): budget, overrides, code and data hashes
- [Training log](original/train.log) and [training metrics](original/train_log.csv)
- [Final weights](original/ckpt_001600.pt) and [resolved configuration](original/run_config.json)
- [Per-clip results](original/eval/001600.json) and [file checksums](original/SHA256SUMS)
- [Source provenance](PROVENANCE.json): published source files are byte-identical to the files
  that ran, except three machine paths in `code/common.py`. Logs, configuration and receipts
  omit host identifiers and use portable paths. The checkpoint keeps its original bytes; the
  per-clip results keep every value but key each clip by the SHA256 of its name.

To rerun it: `python tools/run_record.py --record 2026-11-26_joint_muon ...` (see the
[reproduction instructions](../../README.md#run-the-current-record)). Uni-SuperMimic code,
including the modified learner in the snapshot, is under its
[upstream license](../../third_party/uni-supermimic/LICENSE).
