"""Opt-in engineering patch. Call after hooks.install(), before constructing learner."""
import hashlib,importlib.util,json,pathlib,sys

def install_fast():
 import unisupermimic.rl.algos.supermimic.learner as lm
 cls=lm.SuperMimicLearner
 if getattr(cls,'_engineering_fast_v1',False):return
 folder=pathlib.Path(__file__).resolve().parent
 receipt=json.loads((folder/'learner_optimized.build.json').read_text())
 assert hashlib.sha256(pathlib.Path(lm.__file__).read_bytes()).hexdigest()==receipt['source_sha256'],'Learner source changed; rebuild/revalidate optimization'
 target=folder/'learner_optimized.py'
 assert hashlib.sha256(target.read_bytes()).hexdigest()==receipt['variant_sha256'],'Variant hash mismatch'
 name='unisupermimic.rl.algos.supermimic.learner_fast_v1'
 spec=importlib.util.spec_from_file_location(name,target); module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 cls.train=module.SuperMimicLearner.train
 cls._mean=staticmethod(module.SuperMimicLearner._mean)
 cls._metric_sum=staticmethod(module.SuperMimicLearner._metric_sum)
 module.SuperMimicLearner=cls
 cls._engineering_fast_v1=True
 return receipt
