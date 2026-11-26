"""SuperMimic policy token DAG v1. Fixed signed time axes; explicit routing."""
from __future__ import annotations
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from unisupermimic.rl.algos.supermimic import networks

SCHEMA = "supermimic.policy.token_dag.v1"
D = 192
HEADS = 4

from specs import validate

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(D,eps=1e-5)
        self.gate=nn.Linear(D,512); self.value=nn.Linear(D,512); self.down=nn.Linear(512,D)
    def forward(self,x):
        z=self.norm(x)
        return x+self.down(F.silu(self.gate(z))*self.value(z))

class TemporalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(D,eps=1e-5)
        self.conv=nn.Conv1d(D,D,3,padding=1,groups=D,bias=True)
    def forward(self,x):
        return x+F.silu(self.conv(self.norm(x).transpose(1,2)).transpose(1,2))

def position(axis):
    u=torch.arange(1,11) if axis=="R" else torch.arange(-9,1)
    phase=u.float()[:,None]/(10000**(torch.arange(0,D,2).float()/D))
    return torch.stack((phase.sin(),phase.cos()),dim=-1).flatten(-2)[None]

class Attention(nn.Module):
    def __init__(self,q_axis,c_axis=None,joint=False):
        super().__init__()
        self.joint=joint
        self.cross=c_axis is not None
        self.nq=nn.LayerNorm(D,eps=1e-5)
        self.nc=nn.LayerNorm(D,eps=1e-5) if self.cross else None
        self.q=nn.Linear(D,D,bias=False); self.k=nn.Linear(D,D,bias=False)
        self.v=nn.Linear(D,D,bias=False); self.o=nn.Linear(D,D,bias=False)
        self.register_buffer("pq",position(q_axis),persistent=False)
        self.register_buffer("pc",position(c_axis or q_axis),persistent=False)
        if joint:
            self.qc=nn.Linear(D,D,bias=False); self.kc=nn.Linear(D,D,bias=False)
            self.vc=nn.Linear(D,D,bias=False); self.oc=nn.Linear(D,D,bias=False)
    @staticmethod
    def split(x):
        return x.reshape(x.shape[0],x.shape[1],HEADS,D//HEADS).transpose(1,2)
    @staticmethod
    def attend(q,k,v):
        out=F.scaled_dot_product_attention(Attention.split(q),Attention.split(k),Attention.split(v),dropout_p=0.0)
        return out.transpose(1,2).reshape(q.shape[0],q.shape[1],D)
    def forward(self,x,c=None):
        a=self.nq(x); b=a if c is None else self.nc(c)
        q=self.q(a+self.pq); k=self.k(b+self.pc); v=self.v(b)
        if self.joint:
            q=torch.cat((q,self.qc(b+self.pc)),1)
            k=torch.cat((self.k(a+self.pq),self.kc(b+self.pc)),1)
            v=torch.cat((self.v(a),self.vc(b)),1)
            y=self.attend(q,k,v)
            return x+self.o(y[:,:10]), c+self.oc(y[:,10:])
        return x+self.o(self.attend(q,k,v))

class TokenDAGPolicy(nn.Module):
    def __init__(self,spec):
        super().__init__()
        info=validate(spec)
        self.spec=spec
        self.obs_norm=networks.RunningNorm(99)
        self.goal_norm=networks.RunningNorm(730)
        self.stem_r=nn.Linear(73,D); self.stem_h=nn.Linear(99,D)
        self.blocks=nn.ModuleDict()
        for n in spec["nodes"]:
            op=n["op"]; key=n["id"]
            if op=="swiglu": module=SwiGLU()
            elif op=="temporal_conv": module=TemporalConv()
            elif op=="self_attention": module=Attention(info["axes"][n["input"]])
            elif op=="cross_attention": module=Attention(info["axes"][n["query"]],info["axes"][n["context"]])
            elif op=="joint_attention": module=Attention("R","H",joint=True)
            elif op=="concat_project": module=nn.Linear(2*D,D)
            else: continue
            self.blocks[key]=module
        self.read=nn.Linear(len(spec["readout"]["sources"])*10*D,D)
        self.action=nn.Linear(D,29)
        with torch.no_grad():
            self.action.weight.mul_(0.01); self.action.bias.zero_()
        actual=sum(p.numel() for p in self.parameters())
        if actual!=info["parameters"]: raise ValueError(f"Parameter audit mismatch {actual} != {info['parameters']}")
    def _forward(self,obs_hist,goal):
        values={"R0":self.stem_r(self.goal_norm(goal).reshape(-1,10,73)),
                "H0":self.stem_h(self.obs_norm(obs_hist))}
        for n in self.spec["nodes"]:
            key=n["id"]; op=n["op"]
            if op=="sum":
                values[key]=sum(values[r] for r in n["inputs"])/math.sqrt(len(n["inputs"]))
            elif op=="concat_project":
                values[key]=self.blocks[key](torch.cat([values[r] for r in n["inputs"]],-1))
            elif op=="cross_attention":
                values[key]=self.blocks[key](values[n["query"]],values[n["context"]])
            elif op=="joint_attention":
                values[key+".R"],values[key+".H"]=self.blocks[key](values[n["r"]],values[n["h"]])
            else: values[key]=self.blocks[key](values[n["input"]])
        v=torch.cat([values[r].flatten(1) for r in self.spec["readout"]["sources"]],-1)
        return self.action(F.silu(self.read(v)))
    def forward(self,obs_hist,goal):
        ctx=torch.autocast(obs_hist.device.type,dtype=networks.AMP_DTYPE) if networks.AMP_DTYPE is not None and obs_hist.device.type in ("cuda","mps") else nullcontext()
        with ctx: return self._forward(obs_hist,goal).float()

def make_policy(architecture):
    if architecture["kind"]=="token_dag": return TokenDAGPolicy(architecture["spec"])
    if architecture["kind"]!="native": raise ValueError("Unknown policy kind")
    return networks.Policy(99,730,hist=10,z_dim=32,num_joints=29,hidden=(2048,1024,512,256))

def architecture_hash(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()
