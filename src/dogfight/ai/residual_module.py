# -*- coding: utf-8 -*-
"""FROZEN-BASE RESIDUAL policy for PPO — the structural fix for the SL->PPO collapse.

MEASURED ROOT (260715-16): the DAgger firing/tracking skill is a KNIFE-EDGE fragile weight
configuration (a 0.001% weight change flips wins); ANY PPO gradient backpropagated through the
policy perturbs it -> directional collapse. Fixing the critic (anti-calibrated -> +0.63) did NOT
stop it, because the gradient itself degrades the fragile base. The ONLY structural fix: never
update the base. This module outputs

    action_mean = FROZEN base_pi(o) + eps * tanh(res_head(o))          (residual is bounded, +-eps)

with the base encoder + pi head + base vf FROZEN, and only a small `res_head` (residual on the raw
obs) and a SEPARATE trainable `res_value` critic learning. The base firing skill therefore CANNOT be
degraded — the gradient only moves the bounded residual. res_head's last layer is zero-initialized so
at iter 0 the module is BYTE-IDENTICAL to the frozen base (residual = 0). ResFiT (arXiv:2509.19301).

DEPLOY: raw-stick per-frame — the deployed action = base_pi(o) + eps*tanh(res_head(o)), computed by
the same module in one forward (no iterative denoising, no chunk), so it is deploy-legal as long as
the deploy loader rebuilds this module class (res_head/res_value weights are in the bundle).

TOP-LEVEL class (no closures) so Ray remote workers import it by path.
"""
from __future__ import annotations

from typing import Any, Dict

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class ResidualPPOTorchRLModule(DefaultPPOTorchRLModule):
    """DefaultPPOTorchRLModule with a FROZEN base policy + a bounded trainable residual + a
    separate trainable value net. Only `res_head.*` and `res_value.*` receive gradients."""

    def setup(self):
        super().setup()
        mc = self.model_config if isinstance(self.model_config, dict) else {}
        self._res_eps = float(mc.get("residual_eps", 0.15))
        # ★ 260719 gov-in-loop leg: mask these residual channels (=0, no nudge, no gradient). Throttle (ch3)
        #   stays owned by the governor (round-2: injecting throttle=brake destroys the finish rail).
        self._res_mask = [int(c) for c in (mc.get("residual_mask_channels") or [])]
        # ★ 260725 P-B TERMINAL GATE: g(o)=ramp(ata<hi)*ramp(rng<hi) multiplies the residual so res_head
        #   acts ONLY in the terminal cone/range and is EXACTLY 0 (output AND gradient) elsewhere -> the
        #   drawer learns terminal aim-settling and STRUCTURALLY cannot erode the champion's approach/merge/
        #   defense behavior. Pure obs arithmetic on the CURRENT frame (obs[0:24]: [9]=coarse ATA deg, [6,7,8]
        #   =dNED) -> identical in train/eval/deploy, rides in the bundle, deploy-legal (attitude+relpos only).
        #   OFF by default (gate=1 everywhere) => byte-identical to the ungated residual. Single-residual only
        #   (Stacked/Triple would gate the FROZEN res1 via super() — do not enable there).
        self._gate_on = bool(mc.get("gate_terminal", False))
        self._gate_ata_hi = float(mc.get("gate_ata_hi", 15.0))
        self._gate_ata_ramp = float(mc.get("gate_ata_ramp", 6.0))
        self._gate_rng_hi = float(mc.get("gate_rng_hi", 1200.0))
        self._gate_rng_ramp = float(mc.get("gate_rng_ramp", 400.0))
        obs_dim = int(self.observation_space.shape[0])
        self._n_act = int(self.action_space.shape[0])
        res_hidden = int(mc.get("residual_hidden", 128))
        val_hidden = int(mc.get("residual_val_hidden", 256))

        # bounded residual on the RAW obs (independent of the frozen base encoder)
        self.res_head = nn.Sequential(
            nn.Linear(obs_dim, res_hidden), nn.Tanh(),
            nn.Linear(res_hidden, self._n_act),
        )
        # ★ zero-init the output layer => residual == 0 at iter 0 => module == frozen base exactly
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)

        # SEPARATE trainable critic (the base vf is on the frozen encoder = anti-calibrated/limited)
        self.res_value = nn.Sequential(
            nn.Linear(obs_dim, val_hidden), nn.ReLU(),
            nn.Linear(val_hidden, val_hidden), nn.ReLU(),
            nn.Linear(val_hidden, 1),
        )

        # ★ FREEZE the base: encoder + pi + (base) vf never receive gradients.
        for name in ("encoder", "pi", "vf"):
            mod = getattr(self, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)

    # ── residual injection ───────────────────────────────────────────────────
    def _terminal_gate(self, obs):
        """g(o) = ramp(ata<ata_hi) * ramp(rng<rng_hi) in [0,1], from the CURRENT frame (obs[0:24]).
        None when OFF (caller skips the multiply). Non-negative multiplier -> zeroes residual output AND
        gradient outside the terminal zone."""
        if not self._gate_on:
            return None
        ata = torch.abs(obs[..., 9]) * 180.0                       # o[9]=normalize(ata,-180,180) -> deg (no 10deg sat)
        dn, de, dd = obs[..., 6] * 15000.0, obs[..., 7] * 15000.0, obs[..., 8] * 8000.0
        rng = torch.sqrt(dn * dn + de * de + dd * dd)
        g_ata = torch.clamp((self._gate_ata_hi - ata) / self._gate_ata_ramp, 0.0, 1.0)
        g_rng = torch.clamp((self._gate_rng_hi - rng) / self._gate_rng_ramp, 0.0, 1.0)
        return (g_ata * g_rng).unsqueeze(-1)                       # [...,1] broadcasts over action channels

    def _inject_residual(self, out: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        adi = out[Columns.ACTION_DIST_INPUTS]
        mean = adi[..., : self._n_act]
        tail = adi[..., self._n_act :]  # log_std (free_log_std=False layout: concat(mean, log_std))
        res = torch.tanh(self.res_head(batch[Columns.OBS])) * self._res_eps
        g = self._terminal_gate(batch[Columns.OBS])
        if g is not None:                        # ★ terminal gate: residual acts only in the cone/range
            res = res * g
        if self._res_mask:                       # ★ zero masked channels (e.g. throttle) -> no nudge, no gradient
            res = res.clone(); res[..., self._res_mask] = 0.0
        out[Columns.ACTION_DIST_INPUTS] = torch.cat([mean + res, tail], dim=-1)
        return out

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._inject_residual(super()._forward(batch, **kwargs), batch)

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        out = self._inject_residual(super()._forward_train(batch, **kwargs), batch)
        # PPO's train path expects VF preds under the standard key; use the trainable res_value.
        out[Columns.VF_PREDS] = self.res_value(batch[Columns.OBS]).squeeze(-1)
        return out

    def compute_values(self, batch: Dict[str, Any], embeddings: Any = None):
        return self.res_value(batch[Columns.OBS]).squeeze(-1)


class StackedResidualPPOTorchRLModule(ResidualPPOTorchRLModule):
    """A SECOND bounded residual on top of a FULLY-FROZEN champion (base + res_head1 + res_value all
    frozen). The new res_head2 trains in an EMPTY drawer, so a new skill (e.g. 760m tracking) CANNOT
    evict the tenants already stacked in res_head1 (r11 low-alt + p1d1 BC tracking) -- the interference
    that collapsed the single-residual gov leg's 7000m band. Only res_head2 + res_value2 get gradients.
    action_mean = base_pi(o) + eps1*tanh(res_head1(o))  [frozen champion]  + eps2*tanh(res_head2(o))."""

    def setup(self):
        super().setup()   # base frozen; res_head1 + res_value created (parent leaves them trainable)
        mc = self.model_config if isinstance(self.model_config, dict) else {}
        self._res2_eps = float(mc.get("residual2_eps", 0.2))
        self._res2_mask = [int(c) for c in (mc.get("residual2_mask_channels") or [])]
        obs_dim = int(self.observation_space.shape[0])
        res2_hidden = int(mc.get("residual2_hidden", 128))
        val2_hidden = int(mc.get("residual2_val_hidden", 256))
        self.res_head2 = nn.Sequential(
            nn.Linear(obs_dim, res2_hidden), nn.Tanh(),
            nn.Linear(res2_hidden, self._n_act),
        )
        nn.init.zeros_(self.res_head2[-1].weight)   # zero-init => stacked module == frozen champion at iter 0
        nn.init.zeros_(self.res_head2[-1].bias)
        self.res_value2 = nn.Sequential(
            nn.Linear(obs_dim, val2_hidden), nn.ReLU(),
            nn.Linear(val2_hidden, val2_hidden), nn.ReLU(),
            nn.Linear(val2_hidden, 1),
        )
        # ★ FREEZE res_head1 + res_value = the previous residual is now part of the frozen champion.
        for name in ("res_head", "res_value"):
            for p in getattr(self, name).parameters():
                p.requires_grad_(False)

    def _inject_residual(self, out: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        out = super()._inject_residual(out, batch)   # apply res1 (frozen champion residual)
        adi = out[Columns.ACTION_DIST_INPUTS]
        mean = adi[..., : self._n_act]
        tail = adi[..., self._n_act :]
        res2 = torch.tanh(self.res_head2(batch[Columns.OBS])) * self._res2_eps
        if self._res2_mask:
            res2 = res2.clone(); res2[..., self._res2_mask] = 0.0
        out[Columns.ACTION_DIST_INPUTS] = torch.cat([mean + res2, tail], dim=-1)
        return out

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        out = super()._forward_train(batch, **kwargs)   # applies res1+res2 via our _inject_residual
        out[Columns.VF_PREDS] = self.res_value2(batch[Columns.OBS]).squeeze(-1)   # NEW trainable critic
        return out

    def compute_values(self, batch: Dict[str, Any], embeddings: Any = None):
        return self.res_value2(batch[Columns.OBS]).squeeze(-1)


class TripleStackedResidualPPOTorchRLModule(StackedResidualPPOTorchRLModule):
    """A THIRD bounded residual on top of a FULLY-FROZEN stacked champion (base + res_head1 + res_head2 +
    res_value + res_value2 all frozen). res_head3 trains in an EMPTY drawer so a new skill (e.g. 760m
    rear-defense) CANNOT evict the tenants in res_head1 (champion floor) or res_head2 (7000m yoyo climb) --
    same empty-drawer logic that protected res_head1 when res_head2 was added. Only res_head3 + res_value3
    get gradients.
    action_mean = base_pi(o) + eps1*tanh(res1(o)) + eps2*tanh(res2(o)) + eps3*tanh(res_head3(o))."""

    def setup(self):
        super().setup()   # base + res_head1 frozen; res_head2 + res_value2 created (parent leaves them trainable)
        mc = self.model_config if isinstance(self.model_config, dict) else {}
        self._res3_eps = float(mc.get("residual3_eps", 0.2))
        self._res3_mask = [int(c) for c in (mc.get("residual3_mask_channels") or [])]
        obs_dim = int(self.observation_space.shape[0])
        res3_hidden = int(mc.get("residual3_hidden", 128))
        val3_hidden = int(mc.get("residual3_val_hidden", 256))
        self.res_head3 = nn.Sequential(
            nn.Linear(obs_dim, res3_hidden), nn.Tanh(),
            nn.Linear(res3_hidden, self._n_act),
        )
        nn.init.zeros_(self.res_head3[-1].weight)   # zero-init => triple module == frozen (base+res1+res2) at iter 0
        nn.init.zeros_(self.res_head3[-1].bias)
        self.res_value3 = nn.Sequential(
            nn.Linear(obs_dim, val3_hidden), nn.ReLU(),
            nn.Linear(val3_hidden, val3_hidden), nn.ReLU(),
            nn.Linear(val3_hidden, 1),
        )
        # ★ FREEZE res_head2 + res_value2 = the yoyo-climb residual is now part of the frozen champion.
        for name in ("res_head2", "res_value2"):
            for p in getattr(self, name).parameters():
                p.requires_grad_(False)

    def _inject_residual(self, out: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        out = super()._inject_residual(out, batch)   # apply res1 + res2 (both frozen)
        adi = out[Columns.ACTION_DIST_INPUTS]
        mean = adi[..., : self._n_act]
        tail = adi[..., self._n_act :]
        res3 = torch.tanh(self.res_head3(batch[Columns.OBS])) * self._res3_eps
        if self._res3_mask:
            res3 = res3.clone(); res3[..., self._res3_mask] = 0.0
        out[Columns.ACTION_DIST_INPUTS] = torch.cat([mean + res3, tail], dim=-1)
        return out

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        out = super()._forward_train(batch, **kwargs)   # applies res1+res2+res3 via our _inject_residual
        out[Columns.VF_PREDS] = self.res_value3(batch[Columns.OBS]).squeeze(-1)   # NEW trainable critic
        return out

    def compute_values(self, batch: Dict[str, Any], embeddings: Any = None):
        return self.res_value3(batch[Columns.OBS]).squeeze(-1)
