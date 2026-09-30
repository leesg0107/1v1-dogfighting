"""terminal_takeover — the scripted merge teacher + the WIDE-GATE TERMINAL TAKEOVER, as ONE canonical
source shared byte-for-byte by eval (student.eval.diagnose_merge) AND deploy (student.my_submission via
unreal.policies). Same pattern as dogfight.ai.throttle_governor: single source => train==eval==deploy, no drift.

WHY THIS EXISTS (260725 measured): the champion out-flies every opponent (never hit, always in position) but
CANNOT convert — it holds the 1deg gun cone only 0.1-0.5s/episode and killed 2 of 648 measured episodes (0.31%).
Every attempt to put the aim INTO the champion failed: the governor throttle-pocket (P-A, energy is not the
problem), a gated residual RL drawer (P-B, the knife-edge law — any gradient wrecks the fragile terminal aim),
distillation, reward reshaping, and even a full net trained from scratch on terminal rewards (it350: no better).
What works is putting the aim OUTSIDE and handing over EARLY: inside a WIDE terminal zone (ata<20deg &
rng<1500m) the scripted teacher takes ALL FOUR channels, so it builds its own entry geometry AND pocket energy
(throttle ~0.28) instead of inheriting a blown approach. Measured: 39 kills / 264 episodes (14.8%), 0 deaths,
0 crashes, margin better in every cell across 2 seeds x 2 altitudes x 5 opponents.

The DEAD aim-governor (260719) is the near-miss that proves the mechanism: it took over at rng<700 & |az|<30
and only for roll/pitch — too late (the entry was already blown) and too little (the champion kept throttle
0.88 and punched through). The gate width IS the lever.

DEPLOY-LEGAL: pure obs arithmetic on the tac24 base channels the champion already consumes in deploy
(o[0,4,6,7,8,11,12,20,21,22]) — no new wire, same class as GCAS and the throttle governor. o[17-19] are read
only by the teacher_r variants (off here). Measured tolerance to the deploy path's position-differenced
LOS rates: unaffected at 3 deg/s of injected rate noise, degrades at 8 (deploy is ~0.5-2).
ponytail: gate is a hard AND on (ata, rng). If handoff chatter ever shows up, add hysteresis on exit, not a timer.
"""
from __future__ import annotations
import math

import numpy as np

from dogfight.ai.throttle_governor import throttle_governor


# ★ 260720 teacher-R lag-law constants — MIRROR the TOP of probe_scripted_conversion.py (one source of the
#   same law). teacher-R = teacher-v2 + LAG-bias (aim behind the bandit while high-aspect so the nose
#   approaches the six asymptotically instead of overshooting through it) + aspect throttle-cap.
LAG_GAIN = 0.32        # lag bias = LAG_GAIN * cross_deg, points the nose BEHIND the bandit (deg per deg of aspect)
ASPECT_LOW = 22.0      # cross_deg below which the crossing has bled enough to CLOSE (parity w/ probe; unused in iter-1)
ASPECT_HI = 45.0       # above which still high-aspect -> manage range/energy, don't overshoot in
CROSS_DEADBAND = 12.0  # m/s |rvy| below which lag sign is unreliable -> no lag bias (anti-chatter)
ASPECT_THR_CAP = 0.55  # teacher-R aspect throttle-cap: high-aspect => slow so the turn circle tightens
# ★ teacher-R iter-2 YOYO (pre-registered 260720; ACTIVE via teacher_r=="yoyo"). Operator-approved trigger law:
#   smoothed closure > YOYO_CLO_ARM  AND  range > YOYO_RANGE_GATE_M  AND  aspect low  -> arm a CLIMB (nose-up,
#   trade closure into altitude); on closure bleed-off -> reverse to a bounded DIVE. DISABLED below YOYO_MIN_ALT_M
#   (safety: a wrong climb sign dives into the ground). YOYO_CLO_ARM is a 2-point mini-sweep {120,200} (between
#   brake_arm 120 and measured blow-through +266..454), overridable per-call via pn["yoyo_clo_arm"] (--yoyo-clo-arm).
#   YOYO_RANGE_GATE_M is INDEPENDENT of the brake arm_rng. Modifies ONLY pitch (NO terminal roll/aim override =
#   the dead aim-governor, forbidden). The G-T battery judges.
YOYO_CLO_ARM      = 120.0    # smoothed-closure arm threshold m/s (mini-sweep {120,200}); overtake-only by construction
YOYO_RANGE_GATE_M = 2000.0   # INDEPENDENT of brake arm_rng; yoyo only in the APPROACH (range above this), never terminal
YOYO_MIN_ALT_M    = 1500.0   # DISABLED below this own-altitude (safety floor)
YOYO_ASPECT_MAX   = 30.0     # arm only near nose-on / on-the-six (cross_deg below this) = the blow-through geometry
YOYO_SETTLE_MPS   = 50.0     # smoothed closure bled to here -> reverse CLIMB->DIVE
YOYO_CLIMB_PITCH  = -0.6     # NEGATIVE = nose-up firm climb (trade closure into altitude)
YOYO_DIVE_PITCH   = 0.25     # positive = nose-down bounded reversal back onto the target
# ★ teacher-D DEF (rear-defense break, P0 260722; ACTIVE via teacher_r=="yoyoD" = yoyo + DEF single machine).
#   Ports reactive_defender's threat+break (student/selfplay/autopilot_teacher.py:434-482) into RAW-STICK for the
#   merge teacher. FIRES ONLY on a genuine DEEP-rear, non-extending bandit inside gun-development range so it does
#   NOT clobber the offensive REV/INT/TRK (which own the 25..110deg off-nose band).
#   ★ MEMORYLESS by construction (260722 adversarial review): the lineage burned on "hidden-state latch -> bimodal
#   -> memoryless BC averages the two modes" (the brake law; phase01-round2-verdict). So DEF carries NO per-episode
#   latch — the break direction is the INSTANTANEOUS bandit-bearing sign, which makes (a) the defensive SCISSORS
#   reversal emerge for free the instant the bandit crosses my tail (no dead sign-latch to freeze it) and (b) every
#   DEF label a pure function of the obs (unimodal, distillable). The bank EASES to 0 as the bandit nears dead-six
#   (|los_az|->180) so the ±180 bearing wrap cannot flip the roll frame-to-frame (that flip, not a latch, is the
#   only thing that needed smoothing). Break TOWARD the bandit raises aspect + LOS-rate = defeats the gun solution.
DEF_REAR_CONE_DEG   = 110.0   # |los_az| beyond this = bandit in my DEEP rear (REV owns the 60..110 quarter for offense)
DEF_CLOSE_ARM_MPS   = 0.0     # arm when the deep-rear bandit is NOT extending away (closure>=this); merge-cross post-CPA
                              #   opens (closure<0) so this + the rear cone keep DEF off offensive/merge geometry [A1]
DEF_NEAR_M          = 1800.0  # only defend inside this range (where a gun solution develops); beyond = extend/re-engage
DEF_DEADSIX_EASE_DEG = 30.0   # ease the break bank ->0 across the last this-many deg to dead-six (kills ±180 wrap chatter)
DEF_BREAK_BANK      = 78.0    # break-turn bank target (deg) — parity with HUNT/REV energy-turn bank
DEF_THR             = 0.6     # bleed toward corner speed for a min-radius break (SCAS V_CORNER~190 re-expressed)
DEF_HARD_FLOOR_M    = 600.0   # below this own-alt: never command nose-DOWN (bounds the otherwise-unbounded REV/INT/DEF dive)
DEF_CLIMB_FLOOR_M   = 400.0   # below this own-alt: wings-level + climb-out (survival > the maneuver); > env min_altitude 300m
# ★ REV-강화 probe (season-2 P0 source-ladder, 260723; ACTIVE via teacher_r=="yoyoRev"). DEF NO-GO (G-D2) proved the
#   teacher's EXISTING REV is a STRONGER rear-break than any new overlay (latched bank toward bandit, hard pull, full
#   commitment, NO dead-six de-jink). So don't add a skill — STRENGTHEN REV in the deep-rear low-alt cone: keep every
#   winning trait and only TIGHTEN the turn via corner throttle (DEF's 0.6 over-bled BELOW corner = worse; REV's 1.0
#   sits ABOVE corner = large radius; the sweet spot is between). Conditional amplification of an existing skill, not
#   a new one — different failure mechanism from the 9 prior "create-a-skill" NO-GOs. pn['rev_corner_thr'] = mini-sweep.
REV_DEEP_CONE_DEG = 110.0   # |los_az| beyond this = bandit deep in my rear (same cone DEF used)
REV_NEAR_M        = 1800.0  # only strengthen inside gun-development range
REV_LOWALT_M      = 3000.0  # only at low alt (the 760m failure band); above this REV is already fine (7000m no-regression)
REV_CORNER_THR    = 0.75    # corner throttle in the break = tighter radius (knob; DEF 0.6 < corner < REV 1.0)


def _def_action(_o, _phi, _clo_mps, _rng_now):
    """teacher-D rear-defense break in RAW-STICK, MEMORYLESS. Returns (roll, pitch, thr) iff the CURRENT obs shows a
    deep-rear, non-extending bandit inside gun range, else None (so the offensive phase machine runs). No latch: the
    scissors reversal is instantaneous bearing-following, so the label is a pure function of the obs (distillable)."""
    _los_az = float(_o[11]) * 180.0
    if not (abs(_los_az) > DEF_REAR_CONE_DEG and _clo_mps > DEF_CLOSE_ARM_MPS and _rng_now < DEF_NEAR_M):
        return None
    _sign = 1.0 if _los_az >= 0.0 else -1.0                 # break TOWARD the current bandit side; reverses for free on a tail-cross
    _wrap = min(1.0, (180.0 - abs(_los_az)) / DEF_DEADSIX_EASE_DEG)   # 1 for a clear rear bandit, ->0 at dead-six (anti-wrap-chatter)
    _roll  = float(np.clip((_sign * DEF_BREAK_BANK * _wrap - _phi) / 40.0, -1.0, 1.0))
    _bank_frac = min(1.0, abs(_phi) / 70.0)
    _pitch = float(np.clip(-0.6 * _bank_frac, -0.85, 0.0))  # firm PULL into the break (NEG=nose-up), never nose-down
    return _roll, _pitch, DEF_THR


def merge_teacher_action(_o, st, pn=None, brake=False, teacher_r=False):
    """★ 260713c the merge-fight law (teacher v2, close-range el-blend) as a PURE FUNCTION so the
    `--scripted merge` flyer and the `--scripted dagger` LABELER share one source of truth.
    st = {"rev": float} carries the reversal direction latch. Returns (roll, pitch, thr, phase).

    ★ 260718 (STEP-7 teacher-PN sweep): optional pn dict overrides the PN lead gains WITHOUT changing
    the frozen lineage default (pn=None => the exact gains that seeded r9/r11/r12). Keys + defaults:
      int_az 2.5, int_el 1.5, trk_az 1.2, trk_el 0.8, hunt_bank 78.0, rev_bank 78.0.
    The sweep asks: does a HIGHER-PN teacher law convert more kills vs the champion? If yes, that
    variant is the new distill target (the teacher-gap ceiling that BC-skip r13 is chasing lifts)."""
    _p = pn or {}
    _g_int_az = float(_p.get("int_az", 2.5)); _g_int_el = float(_p.get("int_el", 1.5))
    _g_trk_az = float(_p.get("trk_az", 1.2)); _g_trk_el = float(_p.get("trk_el", 0.8))
    _hunt_bank = float(_p.get("hunt_bank", 78.0)); _rev_bank = float(_p.get("rev_bank", 78.0))
    _az, _el, _phi = float(_o[11]) * 180.0, float(_o[12]) * 90.0, float(_o[0]) * 180.0
    _lr_az = float(_o[20]) * 40.0
    _lr_el = float(_o[21]) * 40.0
    _clo_mps = float(_o[22]) * 600.0
    _dnm, _dem, _ddm = float(_o[6]) * 15000.0, float(_o[7]) * 15000.0, float(_o[8]) * 8000.0
    _rng_now = float(np.sqrt(_dnm * _dnm + _dem * _dem + _ddm * _ddm))
    # ★ 260720 teacher-R MODE selector (teacher_r): False/None = off (byte-identical teacher-v2), "lag" = iter-1
    #   LAG-bias + aspect throttle-cap (below), "yoyo" = iter-2 vertical-yoyo (clean teacher-v2 + pitch-only zoom,
    #   after the phase machine). LAG graft (teacher_r!="lag" => _aim_az==_az => INT/TRK byte-identical to v2):
    #   mirrors probe_scripted_conversion._scripted_action — decode the tac24 rel-vel body (obs[17-19]@600, same
    #   channels as autopilot_teacher.py:303-312), form the aspect proxy cross_deg (also used by yoyo), aim BEHIND.
    _rvx, _rvy, _rvz = float(_o[17]) * 600.0, float(_o[18]) * 600.0, float(_o[19]) * 600.0
    _cross_deg = math.degrees(math.atan2(math.hypot(_rvy, _rvz), abs(_rvx) + 1e-6))   # 0=on-six, 90=abeam
    if teacher_r == "lag" and abs(_rvy) > CROSS_DEADBAND:
        _lag_bias = LAG_GAIN * min(_cross_deg, 90.0) * (1.0 if _rvy > 0.0 else -1.0)
    else:
        _lag_bias = 0.0
    _aim_az = _az - _lag_bias   # == _az when teacher_r off (lag_bias forced 0) -> INT/TRK lead term unchanged
    if _rng_now > 1400.0 and abs(_az) > 25.0:
        # ★ 260717 HUNT: far + off-nose (incl. tails-out diverge starts). The REV law was built for the
        #   1-2km post-merge reversal and measured 0/8 conversion from diverge ICs (climb-oscillates,
        #   never closes). Level energy-keeping turn onto the LOS, then INT/TRK run the target down.
        _phase = "HUNT"
        st["rev"] = 0.0
        _bank_tgt = _hunt_bank if _az >= 0.0 else -_hunt_bank
        _roll = float(np.clip((_bank_tgt - _phi) / 40.0, -1.0, 1.0))
        _bank_frac = min(1.0, abs(_phi) / 70.0)
        _pitch = float(np.clip(-0.55 * _bank_frac + np.clip(-_el / 30.0, -0.2, 0.2), -0.9, 0.3))
        _thr = 1.0
    elif abs(_az) > 60.0:
        _phase = "REV"
        if abs(_az) < 172.0 or st["rev"] == 0.0:
            st["rev"] = 1.0 if _az >= 0.0 else -1.0
        _roll = float(np.clip((st["rev"] * _rev_bank - _phi) / 40.0, -1.0, 1.0))
        _pitch = -0.60
        if _rng_now < 1400.0:
            _pitch = float(np.clip(-0.60 - _el / 60.0, -0.95, -0.25))
        _thr = 1.0
    elif _rng_now > 1400.0:
        _phase = "INT"
        st["rev"] = 0.0
        _az_lead = _aim_az + _g_int_az * _lr_az   # ★ teacher-R: _aim_az==_az when off (byte-identical)
        _roll = float(np.clip(_az_lead / 30.0, -1.0, 1.0))
        _bank_frac = min(1.0, abs(_phi) / 70.0)
        _el_lead = _el + _g_int_el * _lr_el
        _el_term = float(np.clip(-_el_lead / 25.0, -0.6, 0.35))
        _pitch = float(np.clip(-0.5 * _bank_frac + _el_term, -0.95, 0.4))
        _thr = 1.0
    else:
        _phase = "TRK"
        st["rev"] = 0.0
        _az_lead = _aim_az + _g_trk_az * _lr_az   # ★ teacher-R: _aim_az==_az when off (byte-identical)
        _roll = float(np.clip(_az_lead / 12.0, -1.0, 1.0))
        _bank_frac = min(1.0, abs(_phi) / 70.0)
        _el_lead = _el + _g_trk_el * _lr_el
        _el_term = float(np.clip(-_el_lead / 10.0, -0.7, 0.45))
        _pitch = float(np.clip(-0.45 * _bank_frac + _el_term, -0.95, 0.45))
        _thr = 0.6 if (_rng_now < 500.0 and _clo_mps > 120.0 and abs(_az) < 30.0) else 1.0
    # ★ 260719 MERGE-BRAKE (probe #1, anchoring hypothesis): the whole lineage wins the angle but
    #   blows through the 152-914m gun band at +266..454 m/s closure (fire-solution frames = 0 across
    #   7 forensic traces). When we have turned in behind (|az| modest) and are inside 1.5 km, GOVERN
    #   the approach closure down to a settle-able +20..50 m/s so the nose can anchor an in-band aim
    #   instead of penetrating. Excess closure -> cut throttle (bleed E); deficit -> keep closing.
    #
    # ★ 260719b MERGE-BRAKE v2 (P0a, vs-r9 regression fix): v1 was UNCONDITIONAL (any range<1500 &
    #   |az|<60) so it capped throttle mid TAIL-CHASE of an EXTENDING opponent (vs r9: FIRE_SOL
    #   19 -> 7). The discriminator is the closure TREND, not its instant value: a TURNING opponent
    #   is OVERTAKEN fast (closure spikes +250..450 = blow-through = brake), a runner is undertaken
    #   slow (closure ~0..40 = must NOT brake or the chase stalls). So: ARM on a fast overtake inside
    #   the band (smoothed closure > brake_arm); HOLD the governor through the decel-settle even as
    #   closure falls; DISARM when the target extends out of band (fleeing) so the chase gets full
    #   throttle back. brake_arm is a pn knob so P0a can sweep it. opt-in (brake=False => the frozen
    #   lineage law is byte-identical; the whole latch lives under `if brake`).
    #   ponytail: closure-EMA latch (2 state keys). If throttle oscillates at the settle boundary,
    #   widen the arm/disarm hysteresis band before adding a phase timer.
    if teacher_r == "lag" and _phase in ("INT", "TRK") and _cross_deg > ASPECT_HI:
        _thr = min(_thr, ASPECT_THR_CAP)   # ★ 260720 teacher-R aspect throttle-cap: high-aspect => slow, tighten the circle.
        #   Confined to INT/TRK (where the LAG pursuit lives) so HUNT's energy-keeping turn + the REV reversal stay
        #   pure teacher-v2 (cleaner attribution + no energy bleed in the far turn). thr=1.0 default in HUNT/REV.
    # ★ 260720 teacher-R iter-2 VERTICAL YOYO (teacher_r=="yoyo"): CLEAN teacher-v2 (NO lag, NO aspect-cap) + a
    #   PITCH-ONLY zoom. On a fast nose-on overtake in the APPROACH (range > gate) arm a firm CLIMB to trade closure
    #   into altitude (defeats the blow-through); as closure bleeds off, reverse to a bounded DIVE so the nose falls
    #   back onto the target. Per-episode latch in st (like the brake). SAFETY: own_alt<MIN_ALT or range<GATE hard-
    #   DISARMS in ANY state (a wrong climb sign dives into the ground); gated to range>GATE so it is NEVER a terminal
    #   override. Own altitude = obs[4]=normalize(ALT,0,15000) -> (o[4]+1)*7500 (observation.py:520 tac16 base).
    #   Modifies ONLY _pitch (roll/aim stay teacher-v2). yoyo state: 0 off, 1 climbing, 2 diving.
    if teacher_r in ("yoyo", "yoyoD", "yoyoRev"):
        _clo_s = 0.7 * float(st.get("yoyo_clo_s", _clo_mps)) + 0.3 * _clo_mps   # EMA (same law as throttle_governor)
        st["yoyo_clo_s"] = _clo_s
        _own_alt = (float(_o[4]) + 1.0) * 7500.0
        _yoyo_arm = float(_p.get("yoyo_clo_arm", YOYO_CLO_ARM))   # per-call override for the {120,200} mini-sweep
        _yo = int(st.get("yoyo", 0))
        if _own_alt < YOYO_MIN_ALT_M or _rng_now < YOYO_RANGE_GATE_M:
            _yo = 0                                              # hard-DISARM (safety floor OR merged inside the gate)
        elif _yo == 0:
            if _clo_s > _yoyo_arm and _cross_deg < YOYO_ASPECT_MAX:
                _yo = 1                                          # fast nose-on overtake in the approach -> CLIMB
        elif _yo == 1:
            if _clo_s < YOYO_SETTLE_MPS:
                _yo = 2                                          # closure spent -> reverse to a DIVE
        elif _yo == 2:
            if _clo_s < 0.0:
                _yo = 0                                          # target extending -> DISARM
        if _yo == 1:
            _pitch = min(_pitch, YOYO_CLIMB_PITCH)               # more nose-up (NEGATIVE = climb)
        elif _yo == 2:
            _pitch = max(_pitch, YOYO_DIVE_PITCH)                # bounded nose-down reversal back onto the target
        st["yoyo"] = _yo
    # ★ 260722 teacher-D DEF (rear-defense break, memoryless) + low-alt guard — teacher_r=="yoyoD" only.
    if teacher_r == "yoyoD":
        _own_alt = (float(_o[4]) + 1.0) * 7500.0
        _def = _def_action(_o, _phi, _clo_mps, _rng_now)
        if _def is not None:
            _roll, _pitch, _thr = _def
            _phase = "DEF"
            st["yoyo"] = 0                          # DEF overrides — cancel any latched yoyo climb/dive
        # low-alt guard (bounds the otherwise-unbounded REV/INT/DEF dive at the 760m band; roll-to-wings-level mirrors
        # the ported source autopilot_teacher.py:476-479 so the LABEL matches the recovery env GCAS actually flies) [A2]
        if _own_alt < DEF_CLIMB_FLOOR_M:
            _pitch = min(_pitch, -0.25)             # gentle climb-out near the ground (survival > the maneuver)
            _roll  = float(np.clip(-_phi / 40.0, -1.0, 1.0))   # roll to wings-level (a banked pull descends; must level)
        elif _own_alt < DEF_HARD_FLOOR_M:
            _pitch = min(_pitch, 0.0)               # never command nose-DOWN below the floor
    # ★ 260723 REV-강화 probe — STRENGTHEN (not replace) the existing REV/HUNT break in the deep-rear low-alt cone.
    if teacher_r == "yoyoRev":
        _own_alt = (float(_o[4]) + 1.0) * 7500.0
        if abs(_az) > REV_DEEP_CONE_DEG and _clo_mps > 0.0 and _rng_now < REV_NEAR_M and _own_alt < REV_LOWALT_M:
            if abs(_az) < 172.0 or st.get("rev", 0.0) == 0.0:
                st["rev"] = 1.0 if _az >= 0.0 else -1.0
            _roll  = float(np.clip((st["rev"] * _rev_bank - _phi) / 40.0, -1.0, 1.0))   # hard bank toward the bandit (REV's latched sign)
            _pitch = float(np.clip(-0.90 - _el / 60.0, -0.98, -0.55))                   # HARDEST pull = tighter turn (no de-jink)
            _thr   = float(_p.get("rev_corner_thr", REV_CORNER_THR))                    # corner throttle = tighter radius (knob)
            _phase = "REVX"
        if _own_alt < DEF_CLIMB_FLOOR_M:
            _pitch = min(_pitch, -0.25)             # climb-out near the ground (reuse the DEF low-alt guard)
        elif _own_alt < DEF_HARD_FLOOR_M:
            _pitch = min(_pitch, 0.0)
    if brake and _phase not in ("DEF", "REVX"):     # DEF/REVX set throttle deliberately — don't let the approach-brake governor override
        _thr = min(_thr, throttle_governor(_o, st, pn))   # ★ 260719 one-source-of-truth governor (also the hybrid)
    return _roll, _pitch, _thr, _phase

# ── the wide-gate terminal takeover ──────────────────────────────────────────
TAKEOVER_ATA_DEG = 20.0     # coarse ATA (obs[9]) gate; obs[16] saturates at 10deg and cannot express this
TAKEOVER_RNG_M = 1500.0     # range gate — wide enough that the teacher owns the ENTRY, not just the last second


def terminal_takeover(obs, st, nn_action, params=None):
    """Inside the terminal zone the scripted merge+brake teacher owns ALL 4 channels; outside, the NN action
    passes through untouched and the teacher latch resets (it re-enters fresh on the next pass).

    params: {"enabled": bool, "ata": deg, "rng": m}. enabled False (default) => nn_action returned unchanged,
    byte-identical to a build without this wrapper. Returns a raw-stick action [roll, pitch, rudder, throttle]
    in POLICY space [-1,1] (the teacher's throttle is SIM [0,1] and is converted here — the 260719b bug class).
    """
    _p = params or {}
    if not _p.get("enabled", False):
        return nn_action
    _o = np.asarray(obs, dtype=np.float32).ravel()
    _rng = float(np.sqrt((_o[6] * 15000.0) ** 2 + (_o[7] * 15000.0) ** 2 + (_o[8] * 8000.0) ** 2))
    _ata = abs(float(_o[9]) * 180.0)
    # ── RE-ENGAGE arm (260726): the champion EXTENDS AWAY and never comes back ────────────────────
    # Measured after the eval harness was put back on the training disengage rule (it had been running
    # with the cut effectively disabled, so this was invisible): defending at 760 m against a pursuer,
    # 12/12 episodes ended `disengaged` at 26-27 s, 8 km out, 0% damage dealt and 3-16% taken, with
    # RE-ENGAGE x0 -- the champion climbs 760 -> 5.2 km and simply leaves. Head-on merges against a
    # non-engaging opponent do the same at 30-50 s. It only stays and fights when the OPPONENT forces
    # the fight (0/12 disengages against the engaging NN members). The competition manual makes this the
    # most expensive failure we own: out-boxing is a judged penalty/loss, and margin/aim/turn work is
    # worth nothing in an episode we walked out of. So when the range is OPENING past `re_rng`, hand all
    # four channels to the same scripted merge teacher the terminal gate already uses -- it commits to a
    # bank-75 turn toward the bandit and full throttle -- and give control straight back once we are
    # closing again. Closure-gated, so it can only ever fire while we are the ones leaving
    # (overtake-conditional-modifiers law). re_rng 0/absent => OFF => byte-identical.
    _re_rng = float(_p.get("re_rng", 0.0))
    if _re_rng > 0.0:
        _clo = float(_o[22]) * 600.0
        _clo_s = 0.7 * float(st.get("re_clo_s", _clo)) + 0.3 * _clo      # same EMA the governor uses
        st["re_clo_s"] = _clo_s
        if _rng > _re_rng and _clo_s < float(_p.get("re_open", -10.0)):
            st["re_on"] = 1.0
        elif _rng < _re_rng * float(_p.get("re_off_frac", 0.75)) or _clo_s > 0.0:
            st["re_on"] = 0.0
        if float(st.get("re_on", 0.0)) > 0.5:
            # merge_teacher_action's REV branch reads st["rev"] directly (line 147). The terminal gate
            # never trips that branch (it only fires at |az| small), but the re-engage arm fires with the
            # bandit BEHIND us, which is exactly REV -- so seed the latch the caller normally owns.
            st.setdefault("rev", 0.0)
            _rr, _rp, _rt, _ = merge_teacher_action(_o, st, None, False)  # brake OFF: this is a CHASE
            st["armed"] = 1.0        # we are outside the terminal zone by construction; keep the entry arm
            return np.array([float(_rr), float(_rp), 0.0, 1.0], dtype=np.float32)   # full throttle to close
    if _rng >= float(_p.get("rng", TAKEOVER_RNG_M)) or _ata >= float(_p.get("ata", TAKEOVER_ATA_DEG)):
        st.clear()
        st["armed"] = 1.0          # we have been OUTSIDE -> a later entry is a real handoff, not a spawn
        return nn_action
    if not st.get("armed"):
        # ★ 260725b ENTRY-ONLY ARM: the fight STARTED inside the zone (a rear-aspect advantage spawn), so the
        #   champion never flew an approach. Handing over here is not a handoff — it is the teacher flying the
        #   whole engagement, and the teacher is the WORSE pilot (measured rear_aspect vs the scripted threat:
        #   champion +40.3%/2 kills vs teacher-from-spawn +14.7%/0). The takeover only earns its keep on an
        #   entry the champion built, so it stays out until we have been outside the zone at least once.
        return nn_action
    _tr, _tp, _tt, _ = merge_teacher_action(_o, st, None, True)   # brake=True = the merge-brake teacher
    return np.array([float(_tr), float(_tp), 0.0, float(_tt) * 2.0 - 1.0], dtype=np.float32)
