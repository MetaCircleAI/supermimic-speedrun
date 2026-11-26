"""Usage: launch_train.py RUN_NAME GPU ITERATIONS [--smoke] [--resume]"""
from common import *
import subprocess, sys
name, gpu, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
smoke = '--smoke' in sys.argv; resume = '--resume' in sys.argv
run = CAMP / ('smoke' if smoke else 'runs') / name
run.mkdir(parents=True, exist_ok=True)
guard = lock(run / 'train.lock'); assert guard, 'training lease in use'
args = train_args(run, iters); env = environment(gpu, run)
if smoke:
    args = [a for a in args if not a.startswith(('algo.wm_only_iters=', 'algo.log_every=', 'algo.ckpt_every='))]
    args += ['algo.wm_only_iters=0', 'algo.log_every=1', 'algo.ckpt_every=1']
if resume:
    args += ['algo.resume=true', f'algo.load_run={run}', 'algo.checkpoint=last.pt']
write(run / 'command.json', dict(args=args, host=socket.gethostname(), gpu=gpu, iterations=iters, smoke=smoke, resume=resume))
with (run / ('resume.log' if resume else 'train.log')).open('ab', buffering=0) as log:
    p = subprocess.Popen(args, cwd=ROOT / 'source', env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    write(run / 'state.json', dict(status='running', pid=p.pid, host=socket.gethostname(), gpu=gpu, started_at=time.time()))
    rc = p.wait()
completion = read(run / 'training_complete.json', {})
write(run / 'state.json', dict(status=completion.get('status', 'failed') if rc == 0 else 'failed', returncode=rc, completion=completion,
                               host=socket.gethostname(), gpu=gpu, finished_at=time.time()))
raise SystemExit(rc)
