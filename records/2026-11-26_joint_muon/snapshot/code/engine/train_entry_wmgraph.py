import os,time,json,traceback
start=time.monotonic()
from pathlib import Path
import torch
from hooks import install,atomic_json,flush
install(os.environ['SM_ARCH_FILE'])
from install_wmgraph import install_wmgraph
optimization_receipt=install_wmgraph()
atomic_json(Path(os.environ['SM_RUN_DIR'])/'engineering_optimization.json',optimization_receipt)
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
original=SuperMimicLearner._checkpoint
def checkpoint(self,include_optimizers):
    result=original(self,include_optimizers)
    result['training_process_seconds_this_session']=time.monotonic()-start
    result['timing_definition']='wall_time: learner training loop; includes in-loop JIT and checkpoint overhead; excludes env/data initialization, replay warmup and external evaluation'
    return result
SuperMimicLearner._checkpoint=checkpoint
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
try:
    from unisupermimic.scripts.train_supermimic import hydra_main
    hydra_main()
except BaseException as e:
    atomic_json(Path(os.environ['SM_RUN_DIR'])/'error.json',{'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
    raise
finally:
    flush()
    try:
        from torch._dynamo.utils import compile_times
        (Path(os.environ['SM_RUN_DIR'])/'compiler_times.txt').write_text(str(compile_times()))
    except Exception as diagnostic_error:
        print('Compile diagnostics unavailable:', diagnostic_error, flush=True)
