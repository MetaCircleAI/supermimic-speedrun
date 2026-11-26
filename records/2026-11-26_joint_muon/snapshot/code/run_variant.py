"""Usage: run_variant.py VARIANT RUN_NAME GPU ITERATIONS [--smoke] [--eval full|sub10k|none]
Runs in-process like seeded_pipeline.py (cwd = project root), with SM_CAMP=<extra>/<VARIANT>."""
import os, sys, runpy, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
variant, name, gpu, iters = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
smoke = '--smoke' in sys.argv
mode = sys.argv[sys.argv.index('--eval') + 1] if '--eval' in sys.argv else ('smoke' if smoke else 'none')
os.environ['SM_CAMP'] = str(HERE.parent / variant)
sys.path.insert(0, str(HERE))
import common
run = common.CAMP / ('smoke' if smoke else 'runs') / name
sys.argv = ['launch_train.py', name, str(gpu), str(iters)] + (['--smoke'] if smoke else [])
rc = 0
try:
    runpy.run_path(str(common.CODE / 'launch_train.py'), run_name='__main__')
except SystemExit as exc:
    rc = int(exc.code or 0)
done = common.read(run / 'training_complete.json', {})
ok = rc == 0 and done.get('status') == 'completed' and done.get('iteration') == iters
common.write(run / 'pipeline.json', dict(status='trained' if ok else 'train_failed', returncode=rc, at=time.time()))
if not ok: raise SystemExit(1)
if mode != 'none':
    flag = {'smoke': ['--smoke'], 'sub10k': ['--subset10k'], 'full': []}[mode]
    with (run / f'eval_{iters:06d}_{mode}.log').open('ab', buffering=0) as log:
        rc = subprocess.call([common.PY, str(common.CODE / 'eval_one.py'), str(run), str(iters)] + flag,
                             cwd=common.ROOT / 'source', env=common.environment(gpu, run), stdout=log, stderr=subprocess.STDOUT)
    common.write(run / 'pipeline.json', dict(status='done' if rc == 0 else 'eval_failed', eval_mode=mode, returncode=rc, at=time.time()))
raise SystemExit(rc)
