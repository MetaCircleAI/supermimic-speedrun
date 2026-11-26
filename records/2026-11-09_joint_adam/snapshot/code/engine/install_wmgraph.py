"""Hash-checked, opt-in WM CUDA Graph installer; call after hooks.install()."""
import hashlib
import json
from pathlib import Path


def install_wmgraph():
    from install_fast import install_fast
    import unisupermimic.rl.algos.supermimic.learner as lm
    folder = Path(__file__).resolve().parent
    receipt = json.loads((folder / 'wmgraph.build.json').read_text())
    for name, expected in receipt['sha256'].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Engineering package hash mismatch: ' + name)
    cls = lm.SuperMimicLearner
    if getattr(cls, '_engineering_wmgraph_v1', False):
        return receipt
    install_fast()
    from wm_graph import install_wm_graph
    install_wm_graph(cls)
    cls._engineering_wmgraph_v1 = True
    return receipt
