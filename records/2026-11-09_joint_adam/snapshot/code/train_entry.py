"""Speedrun training entry. Timed window: first simulator step of the replay prefill -> final checkpoint
flushed to disk. Imports, data loading and environment construction are untimed; evaluation is separate."""
import os, time, signal, traceback, socket
from pathlib import Path
START = time.monotonic()
import torch
import hooks
from hooks import install, atomic_json, flush
assert os.environ['SM_POLICY_RECOMPUTE'] == '0'
install(os.environ['SM_ARCH_FILE'])
from boundary_fix import install as fix_boundary
fix_boundary()
from install_wmgraph import install_wmgraph
run = Path(os.environ['SM_RUN_DIR'])
atomic_json(run / 'engineering.json', install_wmgraph())
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner as C
from unisupermimic.rl.logging import SuperMimicLogger
requested = False; intentional = False; status = 'completed'; reason = 'iteration budget reached'
TIMER = {}
class Finished(Exception): pass
def stop(*args):
    global requested
    requested = True
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

old_checkpoint = C._checkpoint
def checkpoint(self, include_optimizers):
    ck = old_checkpoint(self, include_optimizers)
    ck['training_process_seconds_this_session'] = time.monotonic() - START
    ck['timing_definition'] = 'speedrun_v1: prefill + learner loop + final checkpoint flush; excludes startup and evaluation'
    ck['policy_recompute'] = False
    return ck
C._checkpoint = checkpoint

old_train = C.train
def train(self, *a, **kw):
    fresh = kw.get('resume') is None and (len(a) == 0 or a[0] is None) and self.iteration == 0
    try:
        if fresh:
            TIMER['start'] = time.monotonic(); t = time.time()
            self.collect(self.cfg.warmup_steps)          # the upstream replay prefill, now inside the timer
            self.wall_time = time.time() - t             # upstream learner clock then also includes it
            TIMER['prefill_seconds'] = self.wall_time; TIMER['prefill_steps'] = self.cfg.warmup_steps
            def skip_upstream_prefill(n, _self=self):   # upstream's first collect(warmup_steps) was just done above
                del _self.collect                       # restore the class method; cfg stays unchanged for resume checks
                if n != _self.cfg.warmup_steps: return _self.collect(n)
            self.collect = skip_upstream_prefill
        result = old_train(self, *a, **kw)
        if fresh:
            flush(); torch.cuda.synchronize()
            TIMER['end'] = time.monotonic()
        return result
    except Finished: raise
    except BaseException as e:
        atomic_json(run / 'root_error.json', dict(error=str(e), traceback=traceback.format_exc(), iteration=self.iteration, wall_time=self.wall_time)); raise
C.train = train

old_log = SuperMimicLogger.log_step
def log(self, iteration, *a, **kw):
    global intentional, reason, status
    result = old_log(self, iteration, *a, **kw); learner = hooks.CURRENT
    if learner is None: return result
    atomic_json(run / 'progress.json', dict(status='training', iteration=iteration, target=learner.cfg.iterations, wall_time=learner.wall_time,
                updated_at=time.time(), max_cuda_allocated=torch.cuda.max_memory_allocated()))
    if requested or (run / 'STOP_TRAIN.json').exists():
        reason = 'safe stop requested'; status = 'paused'
        learner.save(); flush(); intentional = True; raise Finished(reason)
    return result
SuperMimicLogger.log_step = log

torch.set_num_threads(2); rc = 0
try:
    from unisupermimic.scripts.train_supermimic import hydra_main
    hydra_main()
except BaseException as e:
    if not intentional:
        status = 'failed'; rc = 1; atomic_json(run / 'error.json', dict(error=str(e), traceback=traceback.format_exc()))
try: flush()
except BaseException as e:
    status = 'failed'; rc = 1; atomic_json(run / 'flush_error.json', dict(error=str(e)))
learner = hooks.CURRENT
it = getattr(learner, 'iteration', None)
record = dict(status=status, reason=reason, iteration=it, target_iterations=getattr(getattr(learner, 'cfg', None), 'iterations', None),
              learner_wall_time_upstream=getattr(learner, 'wall_time', None), process_seconds=time.monotonic() - START,
              finished_at=time.time(), host=socket.gethostname(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
              max_cuda_allocated=torch.cuda.max_memory_allocated(), policy_recompute=False,
              torch=torch.__version__, device=torch.cuda.get_device_name(0))
if 'start' in TIMER and 'end' in TIMER and status == 'completed':
    record.update(speedrun_seconds=TIMER['end'] - TIMER['start'], prefill_seconds=TIMER['prefill_seconds'],
                  prefill_steps=TIMER['prefill_steps'], final_checkpoint=f'ckpt_{it:06d}.pt',
                  timing_definition='first simulator step of replay prefill -> final checkpoint flushed; startup and evaluation excluded')
atomic_json(run / 'training_complete.json', record)
os._exit(rc)
