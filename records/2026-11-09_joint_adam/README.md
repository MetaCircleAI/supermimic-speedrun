# 27.57 minutes on one H100 (Adam)

On November 9, 2026, the Joint-then-separate token policy (candidate `return_speed_v1_t002`) reached 0.9364642501 normalized
return on the full BONES-SEED library after 2,600 iterations. The training
timer measured 1,654.185 seconds from initial replay prefill through the final
checkpoint write. The evaluation used seed 0 and reported 433 failed clips
out of 129,487. [Original record](RECORD.json)
and [evaluation summary](original/eval/002600.summary.json).

- [Policy specification](snapshot/candidate_spec.json)
- [Timed training entry](snapshot/code/train_entry.py)
- [Final-checkpoint evaluator](snapshot/code/eval_one.py)
- [Fixed-seed launcher](seeded_pipeline.py) and [command with portable paths](command.json)
- [Training log](original/train.log) and [training metrics](original/train_log.csv)
- [Final weights](original/ckpt_002600.pt) and [resolved configuration](original/run_config.json)
- [Per-clip results](original/eval/002600.json) and [original file checksums](original/SHA256SUMS)
- [Source provenance](PROVENANCE.json): published source files against the run's code identity
- [Reproduction instructions](../../README.md#run-the-current-record)

The recipe used Adam for both optimizers, learning rate 0.0008, 100 world-model
warmup iterations, policy EMA 0.98, BF16, compilation and the WM CUDA graph.
The archived launcher sets the algorithm and environment seeds to 1.
The run reused compiler caches from earlier runs of the same code on the same hardware.

The portable entry adapts file locations, Python interpreter, GPU index and
compiler cache location. The checkpoint preserves the original bytes; the per-clip
evaluation keeps every value but keys each clip by the SHA256 of its name. Exported logs and configuration omit host identifiers and use
portable paths. The file checksums cover the published copies.
To rerun it: `python tools/run_record.py --record 2026-11-09_joint_adam ...` (see the
[reproduction instructions](../../README.md#run-the-current-record)). See the
[upstream license](../../third_party/uni-supermimic/LICENSE) for Uni-SuperMimic code, including the
modified learner in the snapshot.
