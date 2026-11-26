"""Experimental CUDA-graph WM update. Replay sampling remains outside capture."""
import torch
from torch import nn

def install_wm_graph(cls):
 def wm_update(self):
  window=self.buffer.sample_windows(self.cfg.wm_batch,self.cfg.wm_window)
  if not hasattr(self,'_wm_graph_lr'):
   self._wm_graph_lr=torch.tensor(self.opt_wm.param_groups[0]['lr'],device=self.device)
   self._wm_graph_stream=torch.cuda.Stream(device=self.device)
   self._wm_graph_warmups=0
  lr=self.opt_wm.param_groups[0]['lr']
  if not torch.is_tensor(lr):self._wm_graph_lr.fill_(lr)
  for group in self.opt_wm.param_groups:
   group['lr']=self._wm_graph_lr;group['capturable']=True
  def step(batch):
   predicted=self._wm_rollout_fn(batch['s'],batch['a'],batch['xi'])
   loss,last_error=self._wm_err_fn(predicted,batch['s1'],batch['mask'])
   self.opt_wm.zero_grad(set_to_none=True)
   loss.backward();nn.utils.clip_grad_norm_(self.wm.parameters(),1.0);self.opt_wm.step()
   mask=batch['mask'][:,-1]
   return loss.detach()/self.cfg.wm_window,(last_error.detach()*mask).sum()/mask.sum().clamp(min=1)
  stream=self._wm_graph_stream;current=torch.cuda.current_stream()
  if self._wm_graph_warmups<3:
   stream.wait_stream(current)
   with torch.cuda.stream(stream):result=step(window)
   current.wait_stream(stream);self._wm_graph_warmups+=1
   return result
  if not hasattr(self,'_wm_graph'):
   self._wm_graph_inputs={k:v.clone() for k,v in window.items()}
   torch.cuda.synchronize()
   first=next(iter(self.wm.parameters()));before=float(self.opt_wm.state[first]['step'])
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):self._wm_graph_outputs=step(self._wm_graph_inputs)
   self._wm_graph=graph
   current.wait_stream(stream);graph.replay();torch.cuda.synchronize()
   after=float(self.opt_wm.state[first]['step']);assert after==before+1,(before,after,'capture must not add extra optimizer updates')
  else:
   for k,v in window.items():self._wm_graph_inputs[k].copy_(v)
   self._wm_graph.replay()
  # Keep per-update metric snapshots: later replays overwrite static output buffers.
  return tuple(v.clone() for v in self._wm_graph_outputs)
 cls.wm_update=wm_update
