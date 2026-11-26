"""Fix terminal-only motion bins without changing bin identities or RNG draws."""
import torch

def install():
 from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms as terms
 if getattr(terms.sample_start_frames,'_terminal_bin_fix',False):return
 original=terms.sample_start_frames
 def safe_sample(motion,bin_G,bin_w_avg,gamma,num_samples,rng,**kwargs):
  frames,bins=original(motion,bin_G,bin_w_avg,gamma,num_samples,rng,**kwargs)
  # A state at clip_end has no outgoing transition. Preserve bin probabilities,
  # IDs and RNG consumption; map only that invalid start to the final legal step.
  ends=motion.clip_end[motion.bin_clip[bins]]
  frames=torch.minimum(frames,ends-1)
  return frames,bins
 safe_sample._terminal_bin_fix=True
 terms.sample_start_frames=safe_sample
 return {'version':'terminal_bin_v1','rule':'frames <= sampled clip_end - 1','unchanged':'bin IDs, probabilities and RNG draws','affected':'terminal-only bins and possible upper-edge rounding'}
