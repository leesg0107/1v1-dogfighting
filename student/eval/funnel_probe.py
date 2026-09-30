"""★ 260725 DAMAGE-FUNNEL probe (offline; feeds on diagnose_merge --dump-csv traces).

The question (user, 260725): p1d1+gov wins on margin but never kills and never dominates
(~30% vs scripts, less vs NNs). WHERE is the kill chain broken? Four pre-registered hypotheses:
  H1 opportunity-NARROWNESS  — mutual rear-chase leaves only short/narrow windows (opponent-made)
  H2 opportunity-CREATION    — champion never breaks the flow (no feints/vertical/tempo changes -> few windows)
  H3 turn-QUALITY            — loses/pars the angles war in sustained turns (never earns the entry)
  H4 terminal-CONVERSION     — earns entries but can't cash them (no cone dwell -> graze damage only)
Discriminators (per cell, vs the merge-brake teacher reference = the lineage's kill existence proof):
  windows/min (creation), window duration (narrowness), p1-cone dwell + dmg/window (conversion),
  turn-rate ratio + angular-advantage slope (quality), dmg-by-phase split (1deg vs late wide cones).

Reads:  <prefix>_epN.csv written by diagnose_merge.py --dump-csv (own_hp column optional/older dumps).
Usage:  python student/eval/funnel_probe.py --cell ch_b1_lo='artifacts/funnel/ch_b1_lo/*_ep*.csv' \
            [--cell ...] [--json artifacts/funnel/funnel_report.json]
WEZ model mirrored from experiments/eval_fv.yaml (competition phase schedule, nested lowest-wins):
  p1 t>=0   ata<1deg 152.4-914.4m  coeff 1.0
  p2 t>=100 ata<2deg 152.4-1066.8m coeff 0.3
  p3 t>=150 ata<3deg 152.4-1219.2m coeff 0.1
"""
import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

DT = 0.1
RMIN, R1, R2, R3 = 152.4, 914.4, 1066.8, 1219.2
WIN_MIN_STEPS = 5          # >=0.5s inside R1 to count as a window (filters blow-through single frames)
KILL_HP = 0.005


def _f(row, k):
    v = row.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def load_ep(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 20:
        return None
    keys = ["rng", "closure", "ata", "los_az", "los_el", "own_alt", "tgt_hp", "own_hp",
            "act_throttle", "oN", "oE", "oD", "oYaw", "tN", "tE", "tD", "tYaw"]
    d = {k: np.array([_f(r, k) for r in rows]) for k in keys}
    d["t"] = np.arange(len(rows)) * DT
    return d


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _seg_slices(mask, min_len=1):
    """contiguous True runs as slices"""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append(slice(i, j))
            i = j
        else:
            i += 1
    return out


def _yawrate(yaw_deg, valid):
    """smoothed deg/s from a possibly NaN-headed yaw trace"""
    y = np.where(valid, yaw_deg, np.nan)
    yr = np.full_like(y, np.nan)
    idx = np.where(valid)[0]
    if len(idx) > 11:
        yy = np.degrees(np.unwrap(np.radians(y[idx])))
        r = np.gradient(yy) / DT
        k = np.ones(11) / 11.0                      # ~1.1s box smooth
        r = np.convolve(r, k, mode="same")
        yr[idx] = r
    return yr


def analyze_ep(d):
    n = len(d["t"])
    rng, ata, t = d["rng"], d["ata"], d["t"]
    tgt_hp = d["tgt_hp"]
    own_hp = d["own_hp"]
    dealt = float(max(0.0, 1.0 - np.nanmin(tgt_hp)))
    taken = float(max(0.0, 1.0 - np.nanmin(own_hp))) if np.isfinite(own_hp).any() else math.nan
    kill = bool(np.nanmin(tgt_hp) <= KILL_HP)
    died = bool(np.isfinite(own_hp).any() and np.nanmin(own_hp) <= KILL_HP)

    # damage-dealt events, bucketed by WEZ phase window (coeffs already embedded in hp deltas)
    dmg_step = np.maximum(0.0, -np.diff(tgt_hp, prepend=tgt_hp[0]))
    dmg_p1 = float(dmg_step[t < 100.0].sum())
    dmg_p2 = float(dmg_step[(t >= 100.0) & (t < 150.0)].sum())
    dmg_p3 = float(dmg_step[t >= 150.0].sum())

    # gun-solution occupancy (recomputed from the schedule; nested = any phase active now)
    sol = ((rng >= RMIN) & (rng <= R1) & (ata < 1.0)) \
        | ((t >= 100.0) & (rng >= RMIN) & (rng <= R2) & (ata < 2.0)) \
        | ((t >= 150.0) & (rng >= RMIN) & (rng <= R3) & (ata < 3.0))
    p1_dwell = float((((rng >= RMIN) & (rng <= R1) & (ata < 1.0))).sum() * DT)
    near3 = float((((rng >= RMIN) & (rng <= R1) & (ata < 3.0))).sum() * DT)   # phase-1 band, almost-aimed

    # ── windows: contiguous presence inside R1 (the fight's entries) ──
    inR = rng <= R1
    wins = _seg_slices(inR, WIN_MIN_STEPS)
    w_stats = []
    for s in wins:
        wdmg = float(dmg_step[s].sum())
        w_stats.append({
            "dur_s": (s.stop - s.start) * DT,
            "min_ata": float(np.nanmin(ata[s])),
            "cone1_s": float(((ata[s] < 1.0) & (rng[s] >= RMIN)).sum() * DT),
            "cone3_s": float(((ata[s] < 3.0) & (rng[s] >= RMIN)).sum() * DT),
            "dmg": wdmg,
            "entry_abs_az": float(abs(d["los_az"][s.start])) if np.isfinite(d["los_az"][s.start]) else math.nan,
        })
    dur_min = t[-1] / 60.0 if t[-1] > 0 else math.nan
    conv = [w for w in w_stats if w["dmg"] > 0.005]

    # ── flow / maneuver-variety metrics ──
    validA = np.isfinite(d["oYaw"])
    oyr = _yawrate(d["oYaw"], validA)
    tyr = _yawrate(d["tYaw"], np.isfinite(d["tYaw"]))
    sgn = np.sign(np.where(np.abs(oyr) > 5.0, oyr, np.nan))    # only committed turns count
    sflip = 0
    last = 0.0
    for v in sgn:
        if np.isnan(v):
            continue
        if last != 0.0 and v != last:
            sflip += 1
        last = v
    alt = d["own_alt"]
    vert_std = float(np.nanstd(alt))
    # orbit-lock (merry-go-round): sustained 914-3000m with little net range progress
    drng = np.abs(np.convolve(np.gradient(rng) / DT, np.ones(11) / 11.0, mode="same"))
    orbit = (rng > R1) & (rng < 3000.0) & (drng < 15.0)
    orbit_frac = float(np.mean([np.any(orbit[max(0, i - 50):i + 50]) and orbit[i] for i in range(n)])) if n else 0.0

    # ── turn-quality during mutual turning inside 2.5km (the angles war) ──
    both = validA & np.isfinite(d["tYaw"]) & (rng < 2500.0) & (np.abs(oyr) > 5.0) & (np.abs(tyr) > 5.0)
    own_tr = float(np.nanmean(np.abs(oyr[both]))) if both.any() else math.nan
    tgt_tr = float(np.nanmean(np.abs(tyr[both]))) if both.any() else math.nan
    # angular advantage: (their horizontal ATA to me) - (mine to them); positive = I'm winning angles
    bee = np.degrees(np.arctan2(d["tE"] - d["oE"], d["tN"] - d["oN"]))
    my_ata_h = np.abs(_wrap(bee - d["oYaw"]))
    their_ata_h = np.abs(_wrap((bee + 180.0) - d["tYaw"]))
    adv = their_ata_h - my_ata_h
    adv_rate = math.nan
    m = np.isfinite(adv) & (rng < 2500.0)
    segs = _seg_slices(m, 30)                                  # >=3s stretches
    if segs:
        rates, wts = [], []
        for s in segs:
            x = t[s]; y = adv[s]
            rates.append(np.polyfit(x, y, 1)[0]); wts.append(len(x))
        adv_rate = float(np.average(rates, weights=wts))
    adv_mean = float(np.nanmean(adv[m])) if m.any() else math.nan

    return {
        "dur_s": float(t[-1]), "dealt": dealt, "taken": taken, "kill": kill, "died": died,
        "dmg_p1": dmg_p1, "dmg_p2": dmg_p2, "dmg_p3": dmg_p3,
        "sol_s": float(sol.sum() * DT), "p1_dwell_s": p1_dwell, "near3_s": near3,
        "frac_inR": float(inR.mean()),
        "n_windows": len(wins), "windows_per_min": len(wins) / dur_min if dur_min else math.nan,
        "win_dur_med": float(np.median([w["dur_s"] for w in w_stats])) if w_stats else 0.0,
        "win_conv_frac": len(conv) / len(wins) if wins else math.nan,
        "dmg_per_window": dealt / len(wins) if wins else math.nan,
        "flow_flips_per_min": sflip / dur_min if dur_min else math.nan,
        "vert_std_m": vert_std, "orbit_frac": orbit_frac,
        "own_turn_dps": own_tr, "tgt_turn_dps": tgt_tr,
        "adv_mean_deg": adv_mean, "adv_rate_dps": adv_rate,
        "windows": w_stats,
    }


def agg(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if not a.size:
        return (math.nan, math.nan)
    return (float(a.mean()), float(a.std() / max(1.0, np.sqrt(a.size))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, help="LABEL=GLOB of *_ep*.csv")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    report = {}
    for spec in a.cell:
        label, pat = spec.split("=", 1)
        eps = []
        for p in sorted(glob.glob(pat)):
            d = load_ep(p)
            if d is not None:
                e = analyze_ep(d)
                e["file"] = p
                eps.append(e)
        if not eps:
            print(f"[{label}] no episodes matched {pat}")
            continue
        N = len(eps)
        kills = sum(e["kill"] for e in eps)
        deaths = sum(e["died"] for e in eps)
        g = {k: agg([e[k] for e in eps]) for k in
             ["dealt", "taken", "sol_s", "p1_dwell_s", "near3_s", "frac_inR", "n_windows",
              "windows_per_min", "win_dur_med", "win_conv_frac", "dmg_per_window",
              "flow_flips_per_min", "vert_std_m", "orbit_frac", "own_turn_dps", "tgt_turn_dps",
              "adv_mean_deg", "adv_rate_dps"]}
        dp = np.array([[e["dmg_p1"], e["dmg_p2"], e["dmg_p3"]] for e in eps]).sum(axis=0)
        dtot = dp.sum() if dp.sum() > 0 else 1.0
        allw = [w for e in eps for w in e["windows"]]
        cone1_tot = sum(w["cone1_s"] for w in allw)
        report[label] = {
            "n": N, "kills": kills, "deaths": deaths,
            **{k: {"mean": v[0], "sem": v[1]} for k, v in g.items()},
            "dmg_phase_split": {"p1": float(dp[0] / dtot), "p2": float(dp[1] / dtot), "p3": float(dp[2] / dtot)},
            "windows_total": len(allw), "cone1_total_s": cone1_tot,
        }
        r = report[label]
        print(f"\n==== {label}  (N={N}) ====")
        print(f"  KILLS {kills}/{N}   deaths {deaths}/{N}   dealt {g['dealt'][0]*100:5.1f}±{g['dealt'][1]*100:.1f}%  "
              f"taken {g['taken'][0]*100:5.1f}%   margin {(g['dealt'][0]-(g['taken'][0] if np.isfinite(g['taken'][0]) else 0))*100:+5.1f}%")
        print(f"  창출 H2: windows/min {g['windows_per_min'][0]:4.2f}±{g['windows_per_min'][1]:.2f}   "
              f"in-R1 점유 {g['frac_inR'][0]*100:4.1f}%   flow-flips/min {g['flow_flips_per_min'][0]:4.1f}   "
              f"vert σ {g['vert_std_m'][0]:5.0f}m   orbit-lock {g['orbit_frac'][0]*100:4.1f}%")
        print(f"  협소 H1: window dur(med) {g['win_dur_med'][0]:4.1f}s")
        print(f"  전환 H4: p1-cone dwell {g['p1_dwell_s'][0]:5.2f}s/ep   near3 {g['near3_s'][0]:5.2f}s/ep   "
              f"sol {g['sol_s'][0]:5.2f}s/ep   conv-window frac {g['win_conv_frac'][0]*100:4.0f}%   "
              f"dmg/window {g['dmg_per_window'][0]*100:4.1f}%p")
        print(f"  턴질 H3: own {g['own_turn_dps'][0]:4.1f} vs tgt {g['tgt_turn_dps'][0]:4.1f} deg/s   "
              f"ang-adv {g['adv_mean_deg'][0]:+5.1f}deg   adv-slope {g['adv_rate_dps'][0]:+5.2f}deg/s")
        print(f"  위상별 dmg: p1(1°콘) {r['dmg_phase_split']['p1']*100:3.0f}%  "
              f"p2(2°) {r['dmg_phase_split']['p2']*100:3.0f}%  p3(3°) {r['dmg_phase_split']['p3']*100:3.0f}%")

    if a.json:
        for c in report.values():
            pass
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2, default=float))
        print(f"\n★ report -> {a.json}")


if __name__ == "__main__":
    main()
