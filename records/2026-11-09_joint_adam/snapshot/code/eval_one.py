"""Usage: eval_one.py RUN_DIR STEP [--smoke]. Full-library (129,487 clips) or frozen 256-clip smoke evaluation."""
from common import *
import sys, hashlib, traceback
import torch
torch.set_num_threads(2)
from hooks import install
install()
from boundary_fix import install as fix_boundary
fix_boundary()
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env, evaluate
run = Path(sys.argv[1]); step = int(sys.argv[2]); smoke = '--smoke' in sys.argv
out = run / 'eval'
ckpt = run / f'ckpt_{step:06d}.pt'; receipt = read(ckpt.with_suffix('.pt.receipt.json'))
assert receipt['bytes'] == ckpt.stat().st_size
started = time.time()
try:
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    if smoke:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from unilab.training import ensure_registries
        from unilab.base.config_adapter import create_env
        from unisupermimic.scripts.train_supermimic import build_env_override
        import unisupermimic
        ensure_registries()
        with initialize_config_dir(version_base='1.3', config_dir=str(Path(unisupermimic.__file__).parent / 'conf/supermimic')):
            cfg = compose(config_name='config', overrides=['task=g1_super_mimic/mjwarp'])
        OmegaConf.update(cfg, 'env.commands.motion.motion_files', [str(ROOT / 'data/eval_subset_256_seed0.pt')])
        env = create_env(cfg, num_envs=256, env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device='cuda'))
    else:
        env = make_eval_env(run, backend='mjwarp', num_envs=16384)
    ck = torch.load(ckpt, map_location=env.device, weights_only=True)
    assert ck['iteration'] == step and ck['experiment_identity'] == read(run / 'identity.json')
    assert ck['policy_recompute'] is False
    result = evaluate(env, ck, device=env.device)
    count = len(result['per_clip']); assert count == (256 if smoke else 129487)
    result.update(checkpoint_iteration=step, training_wall_seconds=ck['wall_time'], evaluation_wall_seconds=time.time() - started,
                  evaluation_scope='smoke256' if smoke else 'full_library', n_clips=count, failure_rate=result['n_failed'] / count,
                  experiment_identity=ck['experiment_identity'], checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  eval_completed_at=time.time(), evaluator_version='infra_v3_terminal_bin_fix', eval_seed=0)
    write(out / f'{step:06d}.json', result)
    summary = {k: v for k, v in result.items() if k not in ('per_clip', 'failed')}
    write(out / f'{step:06d}.summary.json', summary)
    print(json.dumps(summary), flush=True); os._exit(0)
except BaseException as e:
    write(out / f'{step:06d}.error.json', dict(error=str(e), traceback=traceback.format_exc(), at=time.time()))
    traceback.print_exc(); sys.stderr.flush(); os._exit(1)
