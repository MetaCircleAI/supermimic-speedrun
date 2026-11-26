# Baseline: Uni-SuperMimic MLP policy, ~65 minutes

The original Uni-SuperMimic policy (MLP reference encoder to a 32-dimensional code, then an
MLP decoder) with the Adam recipe of its BONES-SEED release: learning rate 4e-4, policy EMA
0.995, 500 world-model-only iterations, algorithm and environment seed 1, on one H100 with the
same BF16, compilation and WM CUDA-graph engineering as the records
([engineering.json](engineering.json)).

This was a 12,000-iteration run that exported a checkpoint every 100 iterations. Selected
checkpoints were evaluated on all 129,487 clips with the deterministic Uni-SuperMimic evaluator
(16,384 environments); [full_library_evals.csv](full_library_evals.csv) lists every one. The
first of them above 0.9363 is iteration 6,400:

| Iteration | Learner time | Full-library return | Failed clips |
| ---: | ---: | ---: | ---: |
| 6,100 | 62.4 min | 0.93520 | 406 |
| 6,300 | 64.4 min | 0.93607 | 415 |
| **6,400** | **65.3 min** | **0.93640** | 352 |
| 6,500 | 66.3 min | 0.93704 | 323 |

Time is the upstream learner clock from [train_log.csv](train_log.csv) (`wall_h`), which starts
after the 64-step replay prefill (5–7 s in the records) and includes the frequent checkpoint
exports. This run's code predates the records' terminal-bin sampling patch
(`runtime/boundary_fix.py` in the record snapshots); it was used neither in this run's training
nor in its evaluation. [CROSSING.json](CROSSING.json) gives the exact values and the hash of the
6,400-iteration checkpoint (108 MB, not included). Files: [configuration](run_config.json),
[arguments](args.json), [evaluation summaries](eval/), [per-clip results at 6,400](eval/006400.json)
(clips keyed by the SHA256 of their names),
[checksums](SHA256SUMS).
