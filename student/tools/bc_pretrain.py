"""bc_pretrain — D1 Component B: BC-pretrain the 950 asset on teacher demonstrations (260711).

WHY (the Run-1 Branch-C verdict): PPO-alone cannot BUILD the acquisition turn — sigma-1.22 sampling
erosion destroys the marginal skill it starts with (L1 42%->11% while training ON it, the 4th clean
reproduction). The injection paradigm: put the teacher's acquisition behavior INTO the weights
directly (this script), then polish with anchored, sigma-lowered PPO (R3, separate).

DESIGN (wf_cfdbcfc3 amendments):
  - Init FROM the 950 bundle's own weights (keeps critic + log_std head + all proven skills).
  - MSE on the MEAN half of ACTION_DIST_INPUTS only (layout = concat(means, log_stds); the log_std
    output columns get no gradient; trunk drift is accepted -> critic-repair phase at R3).
  - Targets = EXECUTED post-GCAS actions (the roller logged them; sampled != executed on 26-40%).
  - Sample weights: ACQ (teacher-flown) frames 1.0; MODULE (self-demonstration/retention) frames
    --ret-weight (anchoring); GCAS-flagged frames x --gcas-mult (0.25: don't imitate the override).
  - EPISODE-level val split (frame-level split leaks adjacent frames).
  - Save = the exact lightweight-bundle format (metadata.json copied from the source bundle +
    policy_weights.pkl.gz) + ROUND-TRIP ASSERT (reload -> forward parity on a fixed batch) +
    weight checksum (the checkpoint_io verifier passes on wholesale mismatch by design — assert here).

Run:
  python student/tools/bc_pretrain.py --bundle-dir artifacts/models/team01/leadturn/bundle_000950 \
    --datasets 'artifacts/bc_v1_*.npz' --out artifacts/models/team01/bc_seed/bundle_v1 \
    --epochs 4 --lr 3e-5 --ret-weight 0.5 --gcas-mult 0.25
"""
from __future__ import annotations
import argparse, glob, gzip, json, pickle, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _pad_obs_input(state, new_dim: int):
    """Warm-start a larger-obs net from a smaller-obs seed by ZERO-PADDING the input Linear layer.

    Only the encoder input layer has in_features == seed_obs_dim (< new_dim); every hidden/head
    weight has in_features == hidden (>= 256 > new_dim), so `shape[1] < new_dim` selects the input
    layer alone. New columns = 0 → the net initially IGNORES the appended obs channels = behaviour-
    identical to the seed. If new_dim == seed_obs_dim nothing matches (normal same-dim warm-start).
    Recurses through the (possibly nested) get_state() dict; only touches 2-D float tensors.
    """
    import torch

    def _fix(v):
        if isinstance(v, dict):
            return {k: _fix(x) for k, x in v.items()}
        if torch.is_tensor(v) and v.ndim == 2 and v.shape[1] < new_dim:
            pad = torch.zeros((v.shape[0], new_dim - v.shape[1]), dtype=v.dtype, device=v.device)
            return torch.cat([v, pad], dim=1)
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] < new_dim:
            pad = np.zeros((v.shape[0], new_dim - v.shape[1]), dtype=v.dtype)
            return np.concatenate([v, pad], axis=1)
        return v

    return _fix(state)


def load_datasets(patterns: list[str], drop_tags: list[str], want_yoyo: bool = False):
    files = sorted(set(sum((glob.glob(p) for p in patterns), [])))
    files = [f for f in files if not any(t in Path(f).name for t in drop_tags)]
    if not files:
        raise SystemExit(f"no dataset files matched {patterns}")
    obs, act, ph, gc, epk, mrg, yoy = [], [], [], [], [], [], []
    ep_base = 0
    for f in files:
        d = np.load(f, allow_pickle=False)
        n = len(d["obs"])
        obs.append(d["obs"]); act.append(d["act"]); ph.append(d["phase"]); gc.append(d["gcas"])
        if want_yoyo:   # ★ 260721 --select yoyo: latch from the SIBLING raw_*.npz (same frames, same append order)
            rf = Path(f).with_name(Path(f).name.replace("bc_", "raw_", 1))
            rd = np.load(rf, allow_pickle=False)
            if "yoyo" not in getattr(rd, "files", []):
                raise SystemExit(f"--select yoyo: {rf} has no 'yoyo' key (re-roll with the yoyo-latch roller edit).")
            yy = np.asarray(rd["yoyo"]).ravel()
            if len(yy) != n:
                raise SystemExit(f"--select yoyo: {rf} yoyo len {len(yy)} != {Path(f).name} rows {n} (misaligned dump).")
            yoy.append(yy)
        # per-frame episode ids from ep_marks (cumulative row counts at each episode end)
        marks = d["ep_marks"]
        eid = np.zeros(n, dtype=np.int64)
        prev = 0
        for k, (_, upto) in enumerate(marks):
            eid[prev:upto] = ep_base + k
            prev = int(upto)
        eid[prev:] = ep_base + len(marks)          # tail rows after the last mark (unterminated ep)
        epk.append(eid)
        ep_base = int(eid.max()) + 1
        # ★ 260714 SIL: per-frame episode margin (if the roller saved ep_margins; else NaN = legacy dataset)
        fm = np.full(n, np.nan, dtype=np.float32)
        if "ep_margins" in getattr(d, "files", []) and len(d["ep_margins"]) == len(marks):
            em = np.asarray(d["ep_margins"], dtype=np.float32); prev = 0
            for k, (_, upto) in enumerate(marks):
                fm[prev:upto] = em[k]; prev = int(upto)     # tail (unterminated) left NaN
        mrg.append(fm)
        acq = int((d["phase"] == 0).sum())
        print(f"  {Path(f).name}: rows={n} ACQ={acq} MODULE={n-acq} eps~{len(marks)}"
              f"{' +margins' if np.isfinite(fm).any() else ''}")
    yoyo = np.concatenate(yoy) if want_yoyo else None
    return (np.concatenate(obs), np.concatenate(act), np.concatenate(ph),
            np.concatenate(gc), np.concatenate(epk), np.concatenate(mrg), yoyo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="npz globs from the --dump-bc roller")
    ap.add_argument("--drop", nargs="*", default=[], help="filename tags to exclude (e.g. a45)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ret-weight", type=float, default=0.5)
    ap.add_argument("--gcas-mult", type=float, default=0.25)
    # ★ 260714 SIL (path-1): filtered / advantage-weighted self-imitation of the champion's OWN rollouts.
    #   The roller (diagnose_merge --dump-bc, champion flies) labels frames with the EXECUTED own action
    #   (phase 1) and saves per-episode health-margin. These turn plain BC into an IMPROVEMENT OPERATOR:
    #   the target is the top-margin tail of the agent's own experience, not the capped teacher.
    #   NOTE: self-flown frames are phase 1 (MODULE) — for pure SIL pass --ret-weight 1.0 so they are not
    #   down-weighted. The PPO-side pairing (sigma-relax + anchor->advantage-weighting) is a SEPARATE step.
    ap.add_argument("--margin-floor", type=float, default=None,
                    help="★ SIL: keep ONLY frames from episodes whose final health-margin >= this "
                         "(filtered self-imitation of converted rollouts). Requires ep_margins in the npz.")
    ap.add_argument("--margin-weight", action="store_true",
                    help="★ SIL: multiply each frame weight by clip(episode_margin, 0, 1) "
                         "(advantage-weighted BC / offline SIL return-proxy). Requires ep_margins.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--select", choices=["all", "acq", "anchor", "anchor_thr", "blend", "yoyo"], default="acq",
                    help="★ 260719b checkpoint-selection metric. 'acq'=legacy best-ACQ (WASHES OUT the brake — "
                         "r10 sol-time lesson). 'anchor'=in-band&|az|<60 action MSE; 'anchor_thr'=that subset's "
                         "THROTTLE channel (=the brake); 'blend'=0.5*acq+0.5*anchor_thr. Phase-1 distill uses anchor_thr/blend. "
                         "★ 260721 'yoyo'=PITCH-MSE on yoyo-window (latch>0) frames, GUARDED by --yoyo-guard on all-frame MSE "
                         "(picks the epoch that best fits the sparse ~4% climb WITHOUT the 96% bulk truncating it; needs "
                         "sibling raw_*.npz carrying the 'yoyo' latch key). Anti checkpoint-selection-leakage (r10/round-1).")
    ap.add_argument("--yoyo-guard", type=float, default=0.10,
                    help="★ 260721 --select yoyo: only epochs with all-frame MSE ≤ (min all-frame MSE)·(1+this) are "
                         "eligible; among them pick min yoyo-window pitch-MSE. Prevents climb-overfit at the bulk's expense.")
    ap.add_argument("--obs-mode", default="tactical24",
                    help="obs mode of the DAgger data + trained net. tactical24_h (28) = history-augmented "
                         "(breaks the reversal-direction aliasing). A smaller-obs seed (e.g. 24-dim 950) is "
                         "warm-started by ZERO-PADDING its input layer, so the 950 finish kernel is preserved.")
    # ★ 260719 round-2 LOSS-WEIGHTING: the merge-brake is a SPARSE, throttle-only signal on ANCHOR frames
    #   (in gun-band 152.4-914.4m & |az|<60, ~8.6% of frames). Unweighted MSE-BC averages it out, so
    #   anchor-thr never converges (0.738->0.509 and still descending at 50ep). Two knobs rebalance it:
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="★ round-2: multiply the SAMPLE weight of ANCHOR frames (in-band & |az|<60) by this. "
                         "1.0 = off (exact round-1 behaviour). Try 4-10 to un-dilute the sparse brake signal.")
    ap.add_argument("--thr-weight", type=float, default=1.0,
                    help="★ round-2: up-weight the THROTTLE channel WITHIN the per-frame MSE on ANCHOR frames "
                         "only (the brake IS the throttle dip). 1.0 = off (equal channels). Surgical brake grad.")
    ap.add_argument("--mask-throttle", action="store_true",
                    help="★ 260719 DAgger-2 channel-targeted distill: EXCLUDE the throttle channel from the BC loss "
                         "(roll/pitch/rudder only). round-2 proved the throttle=brake is distill-proof (rail=finish "
                         "kernel; injecting it collapses the gate), so the brake stays the governor's job. The "
                         "warm-start throttle head is preserved. Use --select acq (throttle-agnostic checkpoint).")
    ap.add_argument("--fresh-init", action="store_true",
                    help="★ round-2 DIAGNOSTIC/pivot: skip the r11 warm-start — random policy init (r11 arch). "
                         "Tests whether the r11 trunk basin (not the data) is the brake bottleneck: a fresh net "
                         "reaches anchor-thr 0.11 offline while warm-start floors at 0.42. Loses the r11 finish "
                         "kernel -> MUST be re-validated closed-loop at the gate, never shipped on offline MSE alone.")
    ap.add_argument("--reset-throttle-head", action="store_true",
                    help="★ round-2 ROOT FIX: de-rail the throttle output. The warm-start throttle mean-logit is "
                         "railed >+1 (r11 = always full throttle); the straight-through clip flatlines the loss so "
                         "BC cannot learn the bimodal brake (anchor-thr stuck ~0.51 vs fresh-net 0.11). Resets ONLY "
                         "the throttle-mean row of the final action projection to small init; trunk + roll/pitch "
                         "(finish kernel) stay warm.")
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    from student.selfplay.bundle_module import BundleModuleCache, default_spaces, build_module_from_metadata
    from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle

    obs_space, act_space = default_spaces(a.obs_mode, action_mode="raw_stick")
    n_act = int(act_space.shape[0])
    obs_dim = int(obs_space.shape[0])
    # ★ 260713 build the (possibly larger) target module, then warm-start from the seed bundle. If the seed
    #   has a SMALLER obs (e.g. 24-dim leadturn/950 into a 28-dim tactical24_h net) zero-pad the input layer
    #   so the net initially ignores the new history channels = byte-behaviour-identical to the 950 seed,
    #   then BC learns to use them. Preserves the 950 finish kernel (the ret-weight lever's whole point).
    metadata, seed_weights = load_lightweight_policy_bundle(a.bundle_dir)
    module = build_module_from_metadata(metadata, obs_space, act_space)
    if a.fresh_init:
        # ★ 260719 diagnostic pivot: keep r11's ARCH but random policy init (no warm-start). The r11 trunk
        #   sits in a throttle-blind basin (warm-start floors anchor-thr at 0.42; a fresh net reaches 0.11).
        print("★ FRESH-INIT: random policy init (r11 arch, NO warm-start) — finish kernel must be re-validated at the gate.")
    else:
        seed_weights = _pad_obs_input(seed_weights, obs_dim)
        module.set_state(seed_weights)
        if a.reset_throttle_head:
            # ★ 260719 de-rail the throttle mean output (see --reset-throttle-head help). The final projection
            #   maps trunk -> ACTION_DIST_INPUTS = concat(means[n_act], log_stds[n_act]); throttle mean = row n_act-1.
            #   Reset that row alone to small init so the throttle logit starts in the responsive [-1,1] region.
            sd = module.state_dict()
            head_w = next(k for k, v in sd.items() if v.ndim == 2 and v.shape[0] == 2 * n_act and v.shape[1] > n_act)
            head_b = head_w.rsplit(".", 1)[0] + ".bias"
            thr_row = n_act - 1
            with torch.no_grad():
                sd[head_w][thr_row].normal_(0.0, 0.01)
                if head_b in sd:
                    sd[head_b][thr_row].zero_()
            module.load_state_dict(sd)
            print(f"★ reset throttle head: {head_w} row {thr_row} (+ bias) -> small init (de-rail)")
    module.train()

    print(f"datasets:")
    obs, act, ph, gc, eid, mrg, yoyo = load_datasets(a.datasets, a.drop, want_yoyo=(a.select == "yoyo"))
    assert obs.shape[1] == obs_dim and act.shape[1] == n_act, \
        f"obs/act shape mismatch: data {obs.shape} vs obs_mode '{a.obs_mode}' dim {obs_dim}, n_act {n_act}"
    assert np.abs(act).max() <= 1.0 + 1e-5, "actions outside [-1,1] — throttle inversion broken?"

    # ★ 260714 SIL filter (path-1): drop frames from non-converted episodes so BC imitates ONLY the
    #   agent's own winning rollouts (the improvement operator). Fill-rate is the pre-registered canary.
    has_margin = bool(np.isfinite(mrg).any())
    if a.margin_floor is not None:
        if not has_margin:
            raise SystemExit("--margin-floor set but datasets lack ep_margins — re-roll with the updated "
                             "diagnose_merge --dump-bc (champion flies, records per-episode margin).")
        _ep_all = len(np.unique(eid))
        keep = np.isfinite(mrg) & (mrg >= a.margin_floor)
        _ep_keep = len(np.unique(eid[keep])) if keep.any() else 0
        print(f"★ SIL margin-floor {a.margin_floor:+.3f}: kept {int(keep.sum())}/{len(keep)} frames | "
              f"{_ep_keep}/{_ep_all} episodes (buffer-fill canary)")
        if int(keep.sum()) < 200:
            raise SystemExit(f"margin-floor too strict: only {int(keep.sum())} frames survive (<200). Conversion is "
                             f"barely sampled under the current exploration — lower the floor or SEED the buffer "
                             f"(WEZ IC-curriculum), do NOT reach for LSTM.")
        obs, act, ph, gc, eid, mrg = (obs[keep], act[keep], ph[keep], gc[keep], eid[keep], mrg[keep])
        if yoyo is not None:
            yoyo = yoyo[keep]

    # sample weights: ACQ 1.0 / MODULE ret_weight, GCAS-flag multiplier
    w = np.where(ph == 0, 1.0, a.ret_weight).astype(np.float32)
    w *= np.where(gc == 1, a.gcas_mult, 1.0).astype(np.float32)
    # ★ 260714 SIL advantage-weighting: scale by the episode's health-margin return proxy (offline SIL / AWR).
    if a.margin_weight:
        if not has_margin:
            raise SystemExit("--margin-weight set but datasets lack ep_margins — re-roll with the updated roller.")
        adv = np.clip(np.nan_to_num(mrg, nan=0.0), 0.0, 1.0).astype(np.float32)
        w = w * adv
        print(f"★ SIL margin-weight ON: weight *= clip(margin,0,1) | mean adv {adv.mean():.3f} "
              f"nonzero {float((adv > 0).mean()) * 100:.0f}%")

    # EPISODE-level val split
    eps = np.unique(eid)
    rng = np.random.default_rng(a.seed); rng.shuffle(eps)
    val_eps = set(eps[: max(1, int(len(eps) * a.val_frac))].tolist())
    val_mask = np.isin(eid, list(val_eps))
    tr_idx, va_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    print(f"train frames={len(tr_idx)} (ACQ {int((ph[tr_idx]==0).sum())}) | val frames={len(va_idx)} | eps {len(eps)} (val {len(val_eps)})")

    t_obs = torch.as_tensor(obs, dtype=torch.float32)
    t_act = torch.as_tensor(act, dtype=torch.float32)
    t_w = torch.as_tensor(w, dtype=torch.float32)
    t_yoyo = None
    if a.select == "yoyo":
        t_yoyo = torch.as_tensor(yoyo > 0)
        _nyw_val = int((yoyo[va_idx] > 0).sum())
        print(f"★ --select yoyo: {int((yoyo>0).sum())} yoyo-window frames ({_nyw_val} in val); "
              f"guard = all-frame MSE ≤ (min)·(1+{a.yoyo_guard})")
        if _nyw_val < 50:
            raise SystemExit(f"only {_nyw_val} yoyo-window val frames — too few for stable selection.")

    from ray.rllib.core.columns import Columns
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)

    def forward_means(idx):
        out = module.forward_train({Columns.OBS: t_obs[idx]})
        m = out[Columns.ACTION_DIST_INPUTS][:, :n_act]      # concat(means, log_stds) -> mean half
        # ★ STRAIGHT-THROUGH CLIP: the module emits raw logits (roll rails at +8..50) while targets are
        #   EXECUTED actions = clip(logit) in [-1,1]. Raw-space MSE punishes behaviour-identical frames
        #   (logit +8 vs target +1 -> MSE 49) and torch.clamp alone zero-grads railed-but-WRONG frames.
        #   Forward = clipped (executed space), backward = identity (railed-wrong frames still pull back).
        return m + (m.clamp(-1.0, 1.0) - m).detach()

    t_ph = torch.as_tensor(ph, dtype=torch.int8)
    # ★ 260719b ANCHOR-STATE mask (in gun band 152.4-914.4m AND |los_az|<60) from the tactical24 base
    #   features (obs[6,7,8]=range comps, obs[11]=az — valid for the whole tac24 family). This is where the
    #   merge-brake governs the throttle; --select anchor/anchor_thr picks the checkpoint that keeps it
    #   (best-ACQ washed it out — the r10-distill sol-time regression, twice reproduced).
    _rng = torch.sqrt((t_obs[:, 6] * 15000.0) ** 2 + (t_obs[:, 7] * 15000.0) ** 2 + (t_obs[:, 8] * 8000.0) ** 2)
    _az = (t_obs[:, 11] * 180.0).abs()
    t_anchor = (_rng > 152.4) & (_rng < 914.4) & (_az < 60.0)
    _thr_ch = n_act - 1   # raw-stick [roll,pitch,rudder,throttle] -> throttle is the last channel
    # ★ 260719 round-2 frame up-weight: rebalance the sparse anchor (merge-brake) frames. --anchor-weight
    #   1.0 = exact no-op (t_fw == t_w) so the round-1 path is preserved and validatable.
    t_fw = t_w * torch.where(t_anchor, float(a.anchor_weight), 1.0)
    print(f"★ anchor frames: {int(t_anchor.sum())}/{len(t_anchor)} ({100.0*float(t_anchor.float().mean()):.1f}%)"
          f" | anchor-weight {a.anchor_weight} thr-weight {a.thr_weight}")
    def val_mse():
        """(all/ACQ/MODULE + ★ANCHOR-state action MSE + ANCHOR-THROTTLE MSE). ACQ was the legacy signal;
        anchor* are the merge-brake retention signals (Phase-1 distill selects on these)."""
        module.eval()
        errs = {k: [0.0, 0.0] for k in ("all", "acq", "mod", "anchor", "anchor_thr", "yoyo")}
        with torch.no_grad():
            for i in range(0, len(va_idx), a.batch):
                idx = va_idx[i:i + a.batch]
                m = forward_means(idx)
                e_raw = ((m - t_act[idx]) ** 2).mean(dim=1)
                e = e_raw * t_w[idx]
                errs["all"][0] += float(e.sum()); errs["all"][1] += float(t_w[idx].sum())
                acq_m = (t_ph[idx] == 0)
                errs["acq"][0] += float(e_raw[acq_m].sum()); errs["acq"][1] += int(acq_m.sum())
                errs["mod"][0] += float(e_raw[~acq_m].sum()); errs["mod"][1] += int((~acq_m).sum())
                an_m = t_anchor[idx]
                errs["anchor"][0] += float(e_raw[an_m].sum()); errs["anchor"][1] += int(an_m.sum())
                thr_e = (m[:, _thr_ch] - t_act[idx][:, _thr_ch]) ** 2
                errs["anchor_thr"][0] += float(thr_e[an_m].sum()); errs["anchor_thr"][1] += int(an_m.sum())
                if t_yoyo is not None:                                 # ★ 260721 PITCH-MSE on yoyo-window frames
                    yw = t_yoyo[idx]
                    pit_e = (m[:, 1] - t_act[idx][:, 1]) ** 2          # channel 1 = pitch (raw-stick)
                    errs["yoyo"][0] += float(pit_e[yw].sum()); errs["yoyo"][1] += int(yw.sum())
        module.train()
        return {k: v[0] / max(v[1], 1e-9) for k, v in errs.items()}

    def sel_metric(v):   # ★ 260719b the checkpoint-selection scalar chosen by --select
        return 0.5 * v["acq"] + 0.5 * v["anchor_thr"] if a.select == "blend" else v[a.select]

    v0 = val_mse()
    print(f"val @init: all {v0['all']:.4f} | ACQ {v0['acq']:.4f} | MODULE {v0['mod']:.4f} | "
          f"★ANCHOR {v0['anchor']:.4f} ANCHOR-thr {v0['anchor_thr']:.4f} | select='{a.select}' ({sel_metric(v0):.4f})")
    best_sel, best_state = sel_metric(v0), None
    _yoyo_sel = (a.select == "yoyo")
    def _snap():   # ★ 260723 FULL state_dict snapshot. Was res_head2/res_value2-only (V3): silent no-op on a PLAIN
                   #   bundle (empty dict -> strict=False restore ships the LAST epoch, not the guarded pick) AND
                   #   missed res_head3 on the current TRIPLE-stack champion. Unify to the line-362 best_state form;
                   #   residual bundles get a superset (frozen parts identical every epoch) = no regression.
        return {k: t.detach().clone() for k, t in module.state_dict().items()}
    curves = [(0, float(v0["all"]), float(v0.get("yoyo", float("nan"))))]   # (epoch, all_frame_mse, yoyo_pitch_mse)
    snaps = {0: _snap()} if _yoyo_sel else None
    for ep_i in range(a.epochs):
        perm = tr_idx.copy(); rng.shuffle(perm)
        tot, tot_w = 0.0, 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            m = forward_means(idx)
            sq = (m - t_act[idx]) ** 2                                    # [B, n_act]
            if a.mask_throttle:          # ★ 260719 DAgger-2: roll/pitch-only distill — mask THROTTLE (=brake, which
                sq = sq.clone(); sq[:, _thr_ch] = 0.0   #   round-2 proved distill-proof); the warm-start throttle (finish
                per = sq.sum(dim=1) / max(n_act - 1, 1) #   kernel) is preserved. Brake stays the governor's job.
            elif a.thr_weight != 1.0:    # up-weight THROTTLE channel on anchor frames (surgical brake grad)
                cw = torch.ones_like(sq)
                cw[:, _thr_ch] = torch.where(t_anchor[idx], float(a.thr_weight), 1.0)
                per = (sq * cw).sum(dim=1) / cw.sum(dim=1)                # weighted channel mean (==mean if all 1)
            else:
                per = sq.mean(dim=1)
            fw = t_fw[idx]
            loss_vec = per * fw
            loss = loss_vec.sum() / fw.sum().clamp_min(1e-9)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss_vec.sum()); tot_w += float(fw.sum())
        v = val_mse()
        star = ""
        if _yoyo_sel:
            curves.append((ep_i + 1, float(v["all"]), float(v["yoyo"]))); snaps[ep_i + 1] = _snap()
        elif sel_metric(v) < best_sel:
            best_sel = sel_metric(v); best_state = {k: t.detach().clone() for k, t in module.state_dict().items()}; star = " ★best"
        print(f"epoch {ep_i+1}/{a.epochs}: train {tot/max(tot_w,1e-9):.4f} | ACQ {v['acq']:.4f} MODULE {v['mod']:.4f} "
              f"ANCHOR {v['anchor']:.4f} ANCHOR-thr {v['anchor_thr']:.4f}"
              + (f" | all {v['all']:.4f} YOYO-pitch {v['yoyo']:.4f}" if _yoyo_sel else f" | sel({a.select}) {sel_metric(v):.4f}{star}"))
    _sel_curves = None
    if _yoyo_sel:
        min_all = min(c[1] for c in curves); thr = min_all * (1.0 + a.yoyo_guard)
        cand = [c for c in curves if c[1] <= thr]
        pick = min(cand, key=lambda c: c[2])
        module.load_state_dict(snaps[pick[0]], strict=False)   # restores the picked epoch's FULL state (see _snap; strict=False harmless)
        print(f"★ --select yoyo: min all-frame MSE {min_all:.4f} → guard ≤{thr:.4f} admits {len(cand)}/{len(curves)} epochs; "
              f"CHOSE epoch {pick[0]} (all {pick[1]:.4f}, YOYO-pitch {pick[2]:.4f} vs init {curves[0][2]:.4f})")
        _sel_curves = curves
    elif best_state is not None:
        module.load_state_dict(best_state)   # early-stop restore: ship the best-'--select' epoch, not the last
        print(f"restored best-{a.select} state (sel-metric {best_sel:.4f})")

    # ── save in the exact lightweight-bundle format + round-trip assert ──
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if _sel_curves is not None:
        import csv
        with open(out / "bc_select_curves.csv", "w", newline="") as fc:
            wr = csv.writer(fc); wr.writerow(["epoch", "all_frame_mse", "yoyo_window_pitch_mse"]); wr.writerows(_sel_curves)
        print(f"  ledger: {out/'bc_select_curves.csv'} (both selection curves)")
    # ★ 260713 stamp the TRAINED obs mode/size into the metadata (the seed's copy would keep the source's
    #   e.g. tactical24/24 → the deploy loader _resolve_bundle_observation_size builds a 24-dim module and
    #   set_weights mismatches the 28-dim tactical24_h weights). Update, don't blind-copy.
    _meta = json.loads((Path(a.bundle_dir) / "metadata.json").read_text(encoding="utf-8"))
    _meta.setdefault("metadata", {})
    _meta["metadata"]["obs_mode"] = a.obs_mode
    _meta["metadata"]["observation_size"] = obs_dim
    (out / "metadata.json").write_text(json.dumps(_meta, indent=2))
    weights = module.get_state()
    with gzip.open(out / "policy_weights.pkl.gz", "wb") as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    # provenance note (does not disturb the loader: metadata.json stays loader-compatible)
    (out / "BC_PROVENANCE.json").write_text(json.dumps({
        "source_bundle": str(a.bundle_dir), "datasets": a.datasets, "drop": a.drop,
        "epochs": a.epochs, "lr": a.lr, "ret_weight": a.ret_weight, "gcas_mult": a.gcas_mult,
        "sil_margin_floor": a.margin_floor, "sil_margin_weight": bool(a.margin_weight),
        "select": a.select, "yoyo_guard": (a.yoyo_guard if a.select == "yoyo" else None),
        "anchor_weight": a.anchor_weight, "thr_weight": a.thr_weight,
        "reset_throttle_head": bool(a.reset_throttle_head), "fresh_init": bool(a.fresh_init),
        "mask_throttle": bool(a.mask_throttle)}, indent=2))

    module2 = BundleModuleCache(obs_space, act_space).load(str(out))
    module.eval(); module2.eval()
    probe = torch.as_tensor(np.random.default_rng(0).standard_normal((64, obs_dim)), dtype=torch.float32)
    with torch.no_grad():
        m1 = module.forward_inference({Columns.OBS: probe})[Columns.ACTION_DIST_INPUTS]
        m2 = module2.forward_inference({Columns.OBS: probe})[Columns.ACTION_DIST_INPUTS]
    assert torch.allclose(m1, m2, atol=1e-5), "ROUND-TRIP FAILED: saved bundle != trained module"
    csum = float(sum(p.detach().abs().sum() for p in module.parameters()))
    print(f"saved {out} | round-trip OK | weight checksum {csum:.2f}")


if __name__ == "__main__":
    main()
