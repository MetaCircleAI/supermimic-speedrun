# Current record code

This folder is the starting point for new attempts: the code of the current record
([2026-11-26_joint_muon](../records/2026-11-26_joint_muon/README.md)) and its recipe.

- `recipe.json`: iteration budget and the Hydra overrides on top of the base recipe in `common.py`
  (the 16.08-minute record uses Muon for policy and world model, lr 1.2e-3, 1,600 iterations).
- `candidate_spec.json`: the Joint token policy graph, built by `runtime/policy_dag.py`.
- `train_entry.py`: the timed training entry (timer from the first prefill step to the final
  checkpoint); `eval_one.py`: the full-library evaluator; `pipeline.py`: train, then evaluate.
- `runtime/hooks.py`: policy construction, Muon parameter split, checkpoints and resume;
  `engine/`: the world-model CUDA graph and the learner with fewer host syncs.

`engine/install_fast.py` and `engine/install_wmgraph.py` check the hashes of the files they patch
in; update `learner_optimized.build.json` or `wmgraph.build.json` if you change those files.
Run it with `../run.sh` (see the [main README](../README.md#run-the-current-record)).
