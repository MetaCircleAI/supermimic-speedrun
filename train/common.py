"""SuperMimic Speedrun v1 (2026-11-08): shared paths, environment and arguments."""
import os, json, time, socket, fcntl
from pathlib import Path
ROOT = Path.cwd()
CAMP = Path(os.environ['SM_CAMP']) if os.environ.get('SM_CAMP') else ROOT / 'experiments/speedrun_v1_20261108'
CODE = CAMP / 'code'
PY = 'python'
PROFILE = (json.loads((CAMP / 'plan.json').read_text()).get('profile') if (CAMP / 'plan.json').exists() else None) or 'speedrun_v1_t002_seed1_final_only'

def read(p, default=None):
    try: return json.loads(Path(p).read_text())
    except FileNotFoundError: return default

def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.tmp.{os.getpid()}'); t.write_text(json.dumps(x, indent=2) + '\n'); t.replace(p)

def lock(p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); h = p.open('a')
    try: fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB); return h
    except BlockingIOError: h.close(); return None

def environment(gpu, run):
    plan = read(CAMP / 'plan.json')
    env = {k: v for k, v in os.environ.items() if not k.startswith('SM_')}
    env.update(SM_ROOT=str(CAMP), SM_RUN_DIR=str(run), SM_ARCH_FILE=str(CAMP / 'candidate_spec.json'),
               SM_POLICY_RECOMPUTE='0', SM_DATA_ID=plan['data_sha256'], SM_PROFILE=PROFILE, SM_CODE_ID=plan['code_sha256'],
               HF_HUB_OFFLINE='1', MUJOCO_GL='disable', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               TORCHINDUCTOR_COMPILE_THREADS='2', CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
               PYTHONPATH=f'{CODE}:{CODE}/runtime:{CODE}/engine', TORCHINDUCTOR_CACHE_DIR=str(Path.home() / '.cache/supermimic-speedrun/inductor'))
    return env

def train_args(run, iterations):
    # t002 recipe; only the iteration budget and checkpoint cadence differ.
    return [PY, str(CODE / 'train_entry.py'), 'algo.seed=1', f'algo.iterations={iterations}', 'algo.pi_opt=adam', 'algo.wm_opt=adam',
            'algo.lr=0.0008', 'algo.wm_only_iters=100', 'algo.pi_ema=0.98', 'algo.no_sync=true', 'algo.compile=true', 'algo.amp=bf16',
            'algo.ckpt_every=1000', 'algo.log_every=25', 'algo.log_backend=no_print',
            f'env.commands.motion.motion_files=[{ROOT}/data/bones_seed_compact.pt]', f'training.log_dir={run}']

_base_train_args = train_args
def train_args(run, iterations):
    # 2026-11-26 extra runs: apply plan.json overrides (same key replaces), always pin env.seed=1.
    args = _base_train_args(run, iterations)
    plan = read(CAMP / 'plan.json', {}) or {}
    over = dict(plan.get('overrides', {}))
    over.setdefault('env.seed', 1)
    keys = {a.split('=', 1)[0]: i for i, a in enumerate(args) if '=' in a}
    for k, v in over.items():
        item = f'{k}={str(v).lower() if isinstance(v, bool) else v}'
        if k in keys: args[keys[k]] = item
        else: args.append(item)
    return args
