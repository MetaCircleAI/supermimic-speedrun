"""Dependency-free contracts and exact canonical labeling."""
import json
import hashlib
SCHEMA="supermimic.policy.token_dag.v1"
D=192

def parents(n):
    op=n["op"]
    if op=="cross_attention": return [n["query"],n["context"]]
    if op=="joint_attention": return [n["r"],n["h"]]
    if op in ("sum","concat_project"): return list(n["inputs"])
    return [n["input"]]

def validate(spec):
    if set(spec)!={"schema","nodes","readout"} or spec["schema"]!=SCHEMA:
        raise ValueError("Invalid schema")
    fields={"swiglu":{"input"},"temporal_conv":{"input"},"self_attention":{"input"},
            "cross_attention":{"query","context"},"joint_attention":{"r","h"},
            "sum":{"inputs"},"concat_project":{"inputs"}}
    axes={"R0":"R","H0":"H"}; deps={"R0":[],"H0":[]}
    k=0; sums=0; produced=[]
    params=(73+99)*D+2*D
    macs=10*(73+99)*D
    for i,n in enumerate(spec["nodes"]):
        op=n.get("op")
        if op not in fields or set(n)!={"id","op"}|fields[op] or n["id"]!=f"n{i}":
            raise ValueError("Invalid node fields/id")
        refs=parents(n)
        if not all(r in axes for r in refs): raise ValueError("Forward/unknown reference")
        a=[axes[r] for r in refs]
        if op=="cross_attention" and a[0]==a[1]: raise ValueError("Cross requires opposite time axes")
        if op=="joint_attention" and a!=["R","H"]: raise ValueError("Joint r/h axis mismatch")
        if op in ("sum","concat_project"):
            legal=(2,3) if op=="sum" else (2,)
            if len(refs) not in legal or len(set(refs))!=len(refs) or len(set(a))!=1:
                raise ValueError("Fusion requires distinct sources on same time axis")
        outs=[n["id"]+".R",n["id"]+".H"] if op=="joint_attention" else [n["id"]]
        for out,axis in zip(outs,["R","H"] if op=="joint_attention" else [a[0]]):
            axes[out]=axis; deps[out]=refs; produced.append(out)
        k+=2 if op=="joint_attention" else (0 if op=="sum" else 1)
        if op=="swiglu":
            params+=3*D*512+1024+D+2*D; macs+=3*10*D*512
        elif op=="temporal_conv":
            params+=6*D; macs+=10*D*3
        elif op=="self_attention":
            params+=4*D*D+2*D; macs+=4*10*D*D+2*100*D
        elif op=="cross_attention":
            params+=4*D*D+4*D; macs+=4*10*D*D+2*100*D
        elif op=="joint_attention":
            params+=8*D*D+4*D; macs+=8*10*D*D+2*400*D
        elif op=="concat_project":
            params+=2*D*D+D; macs+=20*D*D
        elif op=="sum": sums+=1
    if set(spec["readout"])!={"sources"}: raise ValueError("Invalid readout")
    src=spec["readout"]["sources"]
    if len(src) not in (1,2) or len(src)!=len(set(src)) or not all(r in axes for r in src):
        raise ValueError("Invalid readout sources")
    live=set()
    def walk(r):
        if r in live: return
        live.add(r)
        for x in deps[r]: walk(x)
    for r in src: walk(r)
    if not set(axes)<=live: raise ValueError("Dead input/node/joint output")
    if k not in range(2,7) or sums>3 or len(spec["nodes"])>9: raise ValueError("Node budget")
    params+=len(src)*10*D*D+D+D*29+29
    macs+=len(src)*10*D*D+D*29
    if params>9116477 or macs>20000000: raise ValueError("Compute budget")
    return {"K":k,"parameters":params,"macs":macs,"axes":axes}


def canonical(spec):
    validate(spec)
    nodes={n["id"]:n for n in spec["nodes"]}
    required={key:{r.split(".")[0] for r in parents(n) if r not in ("R0","H0")} for key,n in nodes.items()}
    best=None
    def ref(r,mapping):
        parts=r.split(".",1)
        if parts[0] in ("R0","H0"): return r
        return mapping[parts[0]]+("." + parts[1] if len(parts)>1 else "")
    def search(done,mapping,built):
        nonlocal best
        if len(done)==len(nodes):
            obj={"schema":SCHEMA,"nodes":built,"readout":{"sources":sorted(ref(r,mapping) for r in spec["readout"]["sources"])}}
            text=json.dumps(obj,sort_keys=True,separators=(",",":"))
            if best is None or text<best: best=text
            return
        for key in sorted(nodes):
            if key in done or not required[key]<=done: continue
            old=nodes[key]; new={"id":f"n{len(done)}","op":old["op"]}
            for field,value in old.items():
                if field in ("id","op"): continue
                new[field]=sorted(ref(r,mapping) for r in value) if isinstance(value,list) else ref(value,mapping)
            search(done|{key},{**mapping,key:new["id"]},built+[new])
    search(set(),{},[])
    return json.loads(best)

def candidate_identity(spec,contract):
    keys=["schema","interfaces","fixed","nodes","limits","readout"]
    arch={k:contract[k] for k in keys}
    arch_text=json.dumps(arch,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    contract_hash=hashlib.sha256(arch_text.encode()).hexdigest()
    obj=canonical(spec)
    payload={"schema":SCHEMA,"architecture_contract_sha256":contract_hash,"spec":obj}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return obj,digest,contract_hash
