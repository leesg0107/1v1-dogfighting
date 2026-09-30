"""Asymmetric log_std clip for the policy head (TRAINING-ONLY parameterization fix).

★ 260705 rail-fix batch. WHY: ray 2.54 clamps the state-dependent log_std SYMMETRICALLY
(heads.py TorchMLPHead._forward: clamp(log_stds, -c, +c)). With c=0.6 the sigma FLOOR is
e^-0.6=0.549 (blocks precision consolidation) while the CEILING e^+0.6=1.822 allows deep
action-saturation (verified: executed saturation 0.62->0.84 while the MEAN railed; the
saturation rail — not the sigma floor — is what erases conditional credit). Asymmetric
bounds [lo,hi]=[-1.5,+0.2] open the precision floor (sigma>=0.223) and cap the noise
ceiling (sigma<=1.221) — the fix the campaign's own yaml notes pre-registered.

WHY A WORKER HOOK: RLModules are built inside LEARNER and ENV-RUNNER processes, so a
driver-side monkeypatch never reaches them. setup() is registered as ray's
worker_process_setup_hook (runtime_env) AND called once in the driver; bounds travel via
env vars so every process patches identically.

DEPLOY-SAFE: deployment inference takes the deterministic MEAN (clip of logits) — log_std
is never sampled at deploy, and this module is never imported there. Value heads are
untouched (clip_log_std is False for them, same guard ray uses).
"""
from __future__ import annotations

import os

_ENV_LO = "DOGFIGHT_LOGSTD_CLIP_LO"
_ENV_HI = "DOGFIGHT_LOGSTD_CLIP_HI"
_PATCHED_FLAG = "_dogfight_asym_logstd_patched"


def apply(lo: float, hi: float) -> None:
    """Patch TorchMLPHead._forward to clamp log_std into [lo, hi] (policy heads only)."""
    import torch
    from ray.rllib.core.models.torch import heads as _heads
    from ray.rllib.utils.annotations import override
    from ray.rllib.core.models.base import Model

    if getattr(_heads.TorchMLPHead, _PATCHED_FLAG, False):
        return
    lo_f, hi_f = float(lo), float(hi)
    if not (lo_f < hi_f):
        raise ValueError(f"asym log_std clip requires lo < hi, got [{lo_f}, {hi_f}]")

    @override(Model)
    def _forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ANN001
        if self.clip_log_std:
            means, log_stds = torch.chunk(self.net(inputs), chunks=2, dim=-1)
            log_stds = torch.clamp(log_stds, lo_f, hi_f)
            return torch.cat((means, log_stds), dim=-1)
        return self.net(inputs)

    _heads.TorchMLPHead._forward = _forward
    setattr(_heads.TorchMLPHead, _PATCHED_FLAG, True)
    print(f"[asym_logstd] policy-head log_std clip patched to [{lo_f}, {hi_f}] "
          f"(sigma in [{2.718281828**lo_f:.3f}, {2.718281828**hi_f:.3f}]) pid={os.getpid()}",
          flush=True)


def setup() -> None:
    """ray worker_process_setup_hook entrypoint: read bounds from env vars, patch if set."""
    lo, hi = os.environ.get(_ENV_LO), os.environ.get(_ENV_HI)
    if lo is not None and hi is not None:
        apply(float(lo), float(hi))
