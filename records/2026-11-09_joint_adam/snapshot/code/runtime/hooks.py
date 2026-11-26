"""Explicit runtime integration for training, evaluation and resumable snapshots."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import atexit, fcntl, hashlib, json, os, random, time
import numpy as np
import torch
from policy_dag import make_policy

ARCH=None
CURRENT=None
WRITER=ThreadPoolExecutor(max_workers=1,thread_name_prefix="snapshot")
FUTURES=[]

def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj,indent=2,default=str)+"\n")
    os.replace(tmp,path)

def flush():
    for f in FUTURES: f.result()
    FUTURES.clear()
atexit.register(flush)

def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().to("cpu",copy=True)
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value

def rng_state():
    n=np.random.get_state()
    return {"python":random.getstate(),"numpy":[n[0],n[1].tolist(),n[2],n[3],n[4]],
            "torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all()}

def restore_rng(r):
    random.setstate(r["python"])
    n=r["numpy"]; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(r["torch_cpu"].cpu())
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in r["torch_cuda"]])

def identity():
    return {"candidate":ARCH["candidate_id"],"data":os.environ.get("SM_DATA_ID","unset"),
            "profile":os.environ.get("SM_PROFILE","adam_ema_bones_8k_v1"),
            "code":os.environ.get("SM_CODE_ID","unset")}

def write_snapshot(path, state):
    # Shared NFS provides the persistence boundary; bound the aggregate writes.
    lockroot=Path(os.environ["SM_ROOT"])/"runtime_locks"
    lockroot.mkdir(exist_ok=True)
    handle=None
    while handle is None:
        for index in range(2):
            f=(lockroot/f"writer{index}.lock").open("a")
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); handle=f; break
            except BlockingIOError: f.close()
        if handle is None: time.sleep(.2)
    try:
        path=Path(path); path.parent.mkdir(exist_ok=True,parents=True)
        temp=path.with_name(path.name+f".tmp.{os.getpid()}")
        torch.save(state,temp)
        os.replace(temp,path)
        atomic_json(path.with_suffix(path.suffix+".receipt.json"),
                    {"iteration":state["iteration"],"bytes":path.stat().st_size,
                     "saved_at":time.time(),"identity":state["experiment_identity"],
                     "resume_level":state.get("resume_level","evaluation_weights")})
    finally:
        fcntl.flock(handle,fcntl.LOCK_UN); handle.close()

def install(architecture_file=None):
    global ARCH,CURRENT
    if architecture_file is not None:
        ARCH=json.loads(Path(architecture_file).read_text())
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
    from unisupermimic.tasks.motion_tracking.g1.supermimic import evaluate
    from unisupermimic.rl.logging import SuperMimicLogger
    cls=SuperMimicLearner
    old_make=cls._make_policy
    def factory(self):
        if ARCH is None or ARCH["kind"]=="native": return old_make(self)
        if self.cfg.pi_opt!="adam": raise ValueError("Token DAG first profile requires Adam")
        return make_policy(ARCH)
    cls._make_policy=factory

    old_eval=evaluate.load_policy
    def eval_factory(env,checkpoint,device=None):
        arch=checkpoint.get("policy_architecture")
        if arch is None or arch["kind"]=="native": return old_eval(env,checkpoint,device)
        from unisupermimic.rl.algos.supermimic import networks
        networks.AMP_DTYPE=torch.bfloat16 if checkpoint["args"].get("amp")=="bf16" else None
        policy=make_policy(arch).to(device or env.device)
        policy.load_state_dict(checkpoint["pi"]); policy.eval()
        return policy
    evaluate.load_policy=eval_factory

    old_init=cls.__init__
    def init(self,*args,**kwargs):
        global CURRENT
        old_init(self,*args,**kwargs); CURRENT=self
        self.resume_history=[]
        atomic_json(self.log_dir/"architecture.json",ARCH or {"kind":"native","candidate_id":"native"})
        atomic_json(self.log_dir/"identity.json",identity())
    cls.__init__=init

    old_checkpoint=cls._checkpoint
    def checkpoint(self,include_optimizers):
        result=old_checkpoint(self,include_optimizers)
        result["policy_architecture"]=ARCH
        result["experiment_identity"]=identity()
        if include_optimizers:
            result["rng_state"]=rng_state()
            result["value_state"]={"mean":self._v_mean,"std":self._v_std,"init":self._v_init}
            size=self.buffer.size
            result["replay"]={"ptr":self.buffer.ptr,"size":size,"T":self.buffer.T,"N":self.buffer.N,
                              "data":{k:v[:size] for k,v in self.buffer.data.items()}}
            result["collector_diagnostic"]={"obs_hist":self.obs_hist,"info":self._state_info()}
            result["metric_history"]=self.stats
            result["best_metrics"]={}
            result["amp_scaler_state"]=None
            result["resume_history"]=self.resume_history
            result["resume_level"]="model_optimizer_rng_replay_value_stats; collector_reset_nonexact"
        return result
    cls._checkpoint=checkpoint

    def save(self,path=None,include_optimizers=True):
        for future in list(FUTURES):
            if future.done(): future.result(); FUTURES.remove(future)
        # Prevent accumulating unbounded full-replay snapshots in RAM.
        if len(FUTURES)>3: flush()
        destination=Path(path) if path else self.log_dir/"last.pt"
        snapshot=cpu_tree(self._checkpoint(include_optimizers))
        FUTURES.append(WRITER.submit(write_snapshot,destination,snapshot))
        return destination
    cls.save=save

    old_load=cls.load
    def load(self,path):
        ckpt=torch.load(path,map_location="cpu",weights_only=True)
        if ckpt.get("experiment_identity")!=identity():
            raise ValueError("Resume identity mismatch")
        cfg_old=ckpt["args"]; cfg_new=asdict(self.cfg)
        allowed={"iterations","log_every","ckpt_every"}
        if any(cfg_old.get(k)!=v for k,v in cfg_new.items() if k not in allowed):
            raise ValueError("Resume recipe mismatch")
        old_load(self,path)
        state=ckpt.get("value_state")
        if state:
            self._v_mean.copy_(state["mean"].to(self.device)); self._v_std.copy_(state["std"].to(self.device))
            self._v_init=state["init"]
        replay=ckpt.get("replay")
        if replay:
            if (replay["T"],replay["N"])!=(self.buffer.T,self.buffer.N):
                raise ValueError("Replay shape identity mismatch")
            for key,v in replay["data"].items(): self.buffer.data[key][:replay["size"]].copy_(v.to(self.device))
            self.buffer.ptr=replay["ptr"]; self.buffer.size=replay["size"]
        self.resume_history=ckpt.get("resume_history",[])+[{"iteration":self.iteration,"level":"collector_reset_nonexact"}]
        restore_rng(ckpt["rng_state"])
        # Simulator/collector internals are freshly initialized; do not restore an
        # observation history that belongs to a different physical state.
        self.obs_hist.zero_()
        self.collect(self.cfg.hist)
        atomic_json(self.log_dir/"resume_receipt.json",{"iteration":self.iteration,
                    "restored":["model_live","model_ema","world_model","critic","optimizers","rng","replay","value_stats"],
                    "not_restored":["simulator_internal_state","collector_trajectory"],
                    "exact":False})
    cls.load=load

    old_log=SuperMimicLogger.log_step
    def log_step(self,iteration,*args,**kwargs):
        result=old_log(self,iteration,*args,**kwargs)
        if CURRENT is not None:
            if iteration%50==0 or iteration==1:
                atomic_json(CURRENT.log_dir/"progress.json",{"status":"training","iteration":iteration,
                       "target":CURRENT.cfg.iterations,"sim_steps":CURRENT.sim_steps,"wall_time":CURRENT.wall_time,
                       "metrics":kwargs,"identity":identity(),"updated_at":time.time()})
            if iteration%1000==0 and iteration%CURRENT.cfg.ckpt_every!=0: CURRENT.save()
        return result
    SuperMimicLogger.log_step=log_step
