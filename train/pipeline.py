import common, json, runpy, subprocess, sys, time
name, gpu, iterations = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
run = common.CAMP / "runs" / name
# Recipe overrides (Muon, learning rate) and env.seed=1 come from plan.json through common.train_args.
sys.argv = ["launch_train.py", name, str(gpu), str(iterations)]
rc = 0
try:
    runpy.run_path(str(common.CODE / "launch_train.py"), run_name="__main__")
except SystemExit as exc:
    rc = int(exc.code or 0) if exc.code is not None else 0
done = common.read(run / "training_complete.json", {})
if rc != 0 or done.get("status") != "completed" or done.get("iteration") != iterations:
    common.write(run / "pipeline.json", dict(status="train_failed", returncode=rc, completion=done, at=time.time()))
    raise SystemExit(1)
common.write(run / "pipeline.json", dict(status="evaluating", at=time.time()))
with (run / "eval.log").open("ab", buffering=0) as log:
    rc = subprocess.call([common.PY, str(common.CODE / "eval_one.py"), str(run), str(iterations)],
                         cwd=common.ROOT / "source", env=common.environment(gpu, run),
                         stdout=log, stderr=subprocess.STDOUT)
summary = common.read(run / "eval" / f"{iterations:06d}.summary.json", {})
record = dict(run=name, iterations=iterations, algo_seed=1, env_seed=1,
              speedrun_seconds=done.get("speedrun_seconds"),
              speedrun_minutes=(done.get("speedrun_seconds") or 0) / 60,
              norm_return=summary.get("norm_return"),
              passes_target=(summary.get("norm_return") or 0) > 0.9363,
              target=0.9363, prefill_seconds=done.get("prefill_seconds"),
              n_failed=summary.get("n_failed"), gpu=gpu,
              device=done.get("device"), evaluation_returncode=rc)
common.write(run / "RECORD.json", record)
common.write(run / "pipeline.json", dict(status="done" if rc == 0 else "eval_failed", returncode=rc, at=time.time()))
raise SystemExit(rc)
