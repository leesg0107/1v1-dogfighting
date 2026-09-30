# -*- coding: utf-8 -*-
"""TRAIN-ONLY anchored PPO learner (R3, 260713 prescription).

WHY (campaign-verified): PPO-alone on a BC/DAgger seed erases the injected skills two
ways — (i) sigma-1.22 sampling erosion pushes the mean off deterministic-only skills
(Run-1 Branch-C, 260711), (ii) the value function inherited from the SOURCE bundle is
stale for the NEW policy's behavior, so early advantages are garbage and the first
updates are destructive. This learner adds two opt-in mechanisms on top of PPO:

1. ANCHOR (aspect-gated mean-MSE to a FROZEN reference bundle): the reference module
   is the R3 seed itself (bc_seed/bundle_v4d2r). Per-state gate on coarse ATA
   (tactical24 obs[9] = normalize(ata, -180, 180)):
       w = clip((ata_zero - |ata_deg|) / (ata_zero - ata_full), 0, 1)
   full anchor at |ATA| <= 15 deg (the 950 precision-tracking/finishing kernel that
   ret-weight 0.30 preserved = the merge kill-chain's finisher), zero at >= 30 deg
   (anchoring the wide band would freeze the turn-away pathology and erase the very
   skills R3 must improve — 260711 roadmap: "ata_full15/zero30").
   The anchor compares CLIPPED means (executed [-1,1] space) with a straight-through
   gradient — same fix as bc_pretrain.py: raw logits rail to +8..50, raw-space MSE
   punishes behaviour-identical frames and clamp alone zero-grads railed-but-wrong ones.
   sigma is deliberately NOT anchored: the reference carries 950's near-ceiling
   log_std, and anchoring it would fight the lowered sigma ceiling (asym_logstd).

2. CRITIC-REPAIR warmup (first N iterations): vf_share_layers=True means the critic
   is a LINEAR head on the SHARED trunk, so vf-gradient through the trunk would drag
   the policy. During repair the encoder AND pi head are frozen (requires_grad=False)
   and the loss is the VF term alone -> the policy is BIT-IDENTICAL through repair
   (asserted at repair exit) while the linear vf head re-fits the new behavior.

INSTRUMENTS (envelope-assert law, 260711 — every new gate ships its own validation):
  - iter-0 PARITY: first loss call asserts live-module weights == reference bundle
    (catches a broken --init-bundle restore before any update). Skip via
    DOGFIGHT_ANCHOR_SKIP_PARITY=1 (resume-from-later-bundle only).
  - GATE ENVELOPE: over the first 5 loss calls the anchor-gate active fraction must
    land strictly inside (0.02, 0.98) — 0.0/1.0 means the obs index reads the wrong
    channel (the obs[16] +-10deg-saturation class of bug). Skip via
    DOGFIGHT_ANCHOR_GATE_NOASSERT=1.
  - REPAIR-EXIT PARITY: at the repair->PPO transition, non-vf params must still be
    exactly the reference (max|delta| < 1e-6), else the freeze failed.

DEPLOY-SAFE: training path only. Inference/restore rebuild the DEFAULT PPO module +
default learner (same pattern as aux_pred_learner); anchor_* model_config keys are
dropped by _apply_bundle_model_config at restore.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict

from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_KL_KEY
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, VF_LOSS_KEY
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModuleID, TensorType

torch, _ = try_import_torch()

_ENV_SKIP_PARITY = "DOGFIGHT_ANCHOR_SKIP_PARITY"
_ENV_GATE_NOASSERT = "DOGFIGHT_ANCHOR_GATE_NOASSERT"
_GATE_PROBE_CALLS = 5


def apply_anchor_to_config(config, anchor_cfg: dict, root) -> Any:
    """Attach anchor knobs to model_config and swap in the anchored learner class.

    Called by train_rllib AFTER build_algorithm_config (so the MLP model_config is
    already set) and BEFORE build_algo(). Follows the aux_pred precedent: knobs ride
    in a plain-dict model_config (extra keys are ignored by the default catalog);
    the learner class is set via .learners(learner_class=...).
    """
    import dataclasses
    from pathlib import Path

    bundle = Path(str(anchor_cfg["bundle"]))
    if not bundle.is_absolute():
        bundle = (Path(root) / bundle).resolve()
    if not (bundle / "policy_weights.pkl.gz").exists():
        raise FileNotFoundError(
            f"[anchor] reference bundle has no policy_weights.pkl.gz: {bundle}"
        )

    mc = config.model_config
    if dataclasses.is_dataclass(mc):
        mc = dataclasses.asdict(mc)
    mc = dict(mc or {})
    mc.update(
        {
            "anchor_enabled": True,
            "anchor_bundle": str(bundle),
            "anchor_coeff": float(anchor_cfg.get("coeff", 1.0)),
            "anchor_ata_full_deg": float(anchor_cfg.get("ata_full_deg", 15.0)),
            "anchor_ata_zero_deg": float(anchor_cfg.get("ata_zero_deg", 30.0)),
            "anchor_obs_index": int(anchor_cfg.get("obs_index", 9)),
            "anchor_obs_scale_deg": float(anchor_cfg.get("obs_scale_deg", 180.0)),
            "anchor_critic_repair_iters": int(anchor_cfg.get("critic_repair_iters", 50)),
        }
    )
    config = config.rl_module(model_config=mc)
    config = config.learners(learner_class=AnchorPPOTorchLearner)
    return config


class AnchorPPOTorchLearner(PPOTorchLearner):
    """PPOTorchLearner + frozen-reference mean anchor + critic-repair warmup."""

    def build(self):
        super().build()
        self._anchor_ref: Dict[ModuleID, Any] = {}
        self._anchor_calls: Dict[ModuleID, int] = {}
        self._anchor_repair_calls: Dict[ModuleID, int] = {}
        self._anchor_gate_probe: Dict[ModuleID, list] = {}
        self._anchor_frozen: Dict[ModuleID, bool] = {}

    # ── helpers ────────────────────────────────────────────────────────────

    def _anchor_mc(self) -> dict:
        mc = self.config.model_config
        return mc if isinstance(mc, dict) else {}

    @staticmethod
    def _drop_const_leaves(weights, marker: str = "_const"):
        if isinstance(weights, dict):
            return {
                k: AnchorPPOTorchLearner._drop_const_leaves(v, marker)
                for k, v in weights.items()
                if not str(k).endswith(marker)
            }
        return weights

    def _calls_per_iter(self, config) -> int:
        tb = getattr(config, "train_batch_size_per_learner", None)
        if not tb:
            tb = getattr(config, "train_batch_size", 0) or 0
        mb = getattr(config, "minibatch_size", None) or tb or 1
        epochs = int(getattr(config, "num_epochs", 1) or 1)
        return max(1, epochs * max(1, math.ceil(float(tb) / float(mb))))

    def _init_ref_for_module(self, module_id: ModuleID, module, config) -> None:
        """Lazy init at the FIRST loss call: by then the driver has applied the
        --init-bundle weights to this learner, so parity is checkable."""
        import copy as _copy

        from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle

        mc = self._anchor_mc()
        bundle = mc["anchor_bundle"]
        _, weights = load_lightweight_policy_bundle(bundle)
        weights = self._drop_const_leaves(weights)

        ref = _copy.deepcopy(module)
        ref.set_state(weights)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self._anchor_ref[module_id] = ref

        # ★ iter-0 PARITY: live learner weights must BE the reference (init_bundle
        #   == anchor bundle for R3). A silent restore failure here poisoned runs
        #   before (2026-06-13 cold-start postmortem) — fail loud, not late.
        live_sd = dict(module.state_dict())
        ref_sd = dict(ref.state_dict())
        worst_key, worst = "", 0.0
        for k, rv in ref_sd.items():
            if k.endswith("_const") or k not in live_sd:
                continue
            d = float((live_sd[k].detach() - rv.detach()).abs().max())
            if d > worst:
                worst_key, worst = k, d
        if worst > 1e-4 and os.environ.get(_ENV_SKIP_PARITY) != "1":
            raise RuntimeError(
                f"[anchor] ITER-0 PARITY FAIL: live module differs from anchor bundle "
                f"({worst_key} max|Δ|={worst:.3g}). init_bundle restore is broken or "
                f"init_bundle != anchor bundle. Set {_ENV_SKIP_PARITY}=1 ONLY for a "
                f"deliberate resume from a later bundle."
            )
        repair_iters = int(mc.get("anchor_critic_repair_iters", 0))
        cpi = self._calls_per_iter(config)
        self._anchor_repair_calls[module_id] = repair_iters * cpi
        print(
            f"[anchor] ref loaded ({bundle}) | ITER-0 PARITY OK max|Δ|={worst:.3g} "
            f"| coeff={mc.get('anchor_coeff')} gate=[full<={mc.get('anchor_ata_full_deg')}deg,"
            f"zero>={mc.get('anchor_ata_zero_deg')}deg] obs[{mc.get('anchor_obs_index')}] "
            f"| critic-repair {repair_iters} iters x {cpi} calls/iter",
            flush=True,
        )

    def _set_repair_freeze(self, module, frozen: bool) -> None:
        """Freeze/unfreeze everything except the vf head (linear on the shared trunk)."""
        for name, p in module.named_parameters():
            if name.startswith("vf.") or ".vf." in name:
                p.requires_grad_(True)
            else:
                p.requires_grad_(not frozen)

    def _repair_exit_parity(self, module_id: ModuleID, module) -> None:
        ref = self._anchor_ref[module_id]
        live_sd = dict(module.state_dict())
        worst_key, worst = "", 0.0
        for k, rv in dict(ref.state_dict()).items():
            if k.endswith("_const") or k not in live_sd:
                continue
            if k.startswith("vf.") or ".vf." in k:
                continue  # the vf head is SUPPOSED to have moved
            d = float((live_sd[k].detach() - rv.detach()).abs().max())
            if d > worst:
                worst_key, worst = k, d
        if worst > 1e-6:
            raise RuntimeError(
                f"[anchor] REPAIR-EXIT PARITY FAIL: non-vf params moved during the "
                f"critic-repair freeze ({worst_key} max|Δ|={worst:.3g}) — freeze broken."
            )
        print(f"[anchor] critic-repair DONE — REPAIR-EXIT PARITY OK (max non-vf |Δ|={worst:.3g})", flush=True)

    # ── loss ───────────────────────────────────────────────────────────────

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: Any,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        mc = self._anchor_mc()
        if not mc.get("anchor_enabled"):
            return super().compute_loss_for_module(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )

        module = self.module[module_id].unwrapped()
        if module_id not in self._anchor_ref:
            self._init_ref_for_module(module_id, module, config)

        calls = self._anchor_calls.get(module_id, 0)
        self._anchor_calls[module_id] = calls + 1
        repair_calls = self._anchor_repair_calls.get(module_id, 0)
        in_repair = calls < repair_calls

        # repair freeze bookkeeping (freeze BEFORE compute_values below so the vf
        # path re-runs the encoder grad-free; unfreeze exactly once at transition).
        if in_repair and not self._anchor_frozen.get(module_id, False):
            self._set_repair_freeze(module, True)
            self._anchor_frozen[module_id] = True
            print(f"[anchor] critic-repair phase ENTERED (calls 0..{repair_calls})", flush=True)
        elif not in_repair and self._anchor_frozen.get(module_id, False):
            self._repair_exit_parity(module_id, module)
            self._set_repair_freeze(module, False)
            self._anchor_frozen[module_id] = False

        obs = batch[Columns.OBS]
        ref = self._anchor_ref[module_id]

        # ── anchor term (computed in BOTH phases; in repair it is log-only) ──
        dist_inputs = fwd_out[Columns.ACTION_DIST_INPUTS]
        n_act = dist_inputs.shape[-1] // 2
        cur_mean = dist_inputs[..., :n_act]
        with torch.no_grad():
            ref_out = ref.forward_train({Columns.OBS: obs})
            ref_mean = ref_out[Columns.ACTION_DIST_INPUTS][..., :n_act].clamp(-1.0, 1.0)
        # straight-through clip (bc_pretrain fix): forward=executed space, backward=identity
        cur_st = cur_mean + (cur_mean.clamp(-1.0, 1.0) - cur_mean).detach()
        se = ((cur_st - ref_mean) ** 2).mean(-1)

        idx = int(mc.get("anchor_obs_index", 9))
        scale = float(mc.get("anchor_obs_scale_deg", 180.0))
        full = float(mc.get("anchor_ata_full_deg", 15.0))
        zero = float(mc.get("anchor_ata_zero_deg", 30.0))
        ata_deg = obs[..., idx].abs() * scale
        w = ((zero - ata_deg) / max(zero - full, 1e-6)).clamp(0.0, 1.0)

        loss_mask = batch.get(Columns.LOSS_MASK)
        if loss_mask is not None:
            w = w * loss_mask.float()
        anchor_loss = (se * w).sum() / w.sum().clamp(min=1.0)
        gate_frac = float((w > 0).float().mean())

        # ★ GATE ENVELOPE assert (first N calls): 0/1 gate fraction == wrong channel.
        probe = self._anchor_gate_probe.setdefault(module_id, [])
        if len(probe) < _GATE_PROBE_CALLS:
            probe.append(gate_frac)
            if len(probe) == 1:
                q = torch.quantile(
                    ata_deg.detach().flatten().float(),
                    torch.tensor([0.1, 0.5, 0.9], device=ata_deg.device),
                )
                print(
                    f"[anchor] gate probe call-1: |ATA| P10/50/90 = "
                    f"{q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} deg | gate_frac={gate_frac:.3f} "
                    f"| anchor_mse={anchor_loss.detach().item():.3g}",
                    flush=True,
                )
            if len(probe) == _GATE_PROBE_CALLS:
                gf = sum(probe) / len(probe)
                if not (0.02 < gf < 0.98) and os.environ.get(_ENV_GATE_NOASSERT) != "1":
                    raise RuntimeError(
                        f"[anchor] GATE ENVELOPE FAIL: mean gate_frac={gf:.4f} over first "
                        f"{_GATE_PROBE_CALLS} calls (expected strictly inside (0.02,0.98)). "
                        f"obs[{idx}] is probably NOT the coarse-ATA channel (obs[16]-class "
                        f"saturation bug). Set {_ENV_GATE_NOASSERT}=1 only with cause."
                    )
                print(f"[anchor] GATE ENVELOPE OK: mean gate_frac={gf:.3f}", flush=True)

        if in_repair:
            # vf-only fit (policy frozen => anchor grad is structurally zero; log it).
            if Columns.LOSS_MASK in batch:
                mask = batch[Columns.LOSS_MASK]
                num_valid = torch.sum(mask)

                def pm_mean(t):
                    return torch.sum(t[mask]) / num_valid

            else:
                pm_mean = torch.mean
            value_fn_out = module.compute_values(batch, embeddings=None)
            vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)
            vf_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf = pm_mean(vf_clipped)
            total = config.vf_loss_coeff * mean_vf
            # ★ repair-phase OBSERVABILITY (else the console shows entropy/KL = n/a for the
            #   whole 50-iter repair window — the operator cannot confirm the policy is frozen
            #   or entropy healthy). Log ENTROPY (from the current dist) and a KL-vs-behavior
            #   (must be ~0 while the policy is frozen = a live console proof of the freeze).
            with torch.no_grad():
                dist_cls = module.get_train_action_dist_cls()
                cur_dist = dist_cls.from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
                mean_ent = pm_mean(cur_dist.entropy())
                beh_cls = module.get_exploration_action_dist_cls()
                beh_dist = beh_cls.from_logits(batch[Columns.ACTION_DIST_INPUTS])
                repair_kl = pm_mean(beh_dist.kl(cur_dist))
            self.metrics.log_dict(
                {
                    VF_LOSS_KEY: mean_vf.detach().item(),
                    ENTROPY_KEY: mean_ent.item(),
                    LEARNER_RESULTS_KL_KEY: repair_kl.item(),
                    "anchor_loss": anchor_loss.detach().item(),
                    "anchor_gate_frac": gate_frac,
                    "anchor_repair_phase": 1.0,
                },
                key=module_id,
                window=1,
            )
            return total

        ppo_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        coeff = float(mc.get("anchor_coeff", 1.0))
        self.metrics.log_dict(
            {
                "anchor_loss": anchor_loss.detach().item(),
                "anchor_gate_frac": gate_frac,
                "anchor_repair_phase": 0.0,
            },
            key=module_id,
            window=1,
        )
        return ppo_loss + coeff * anchor_loss
