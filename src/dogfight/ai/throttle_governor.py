"""throttle_governor — the merge-brake latch as ONE canonical pure function, shared byte-for-byte by
the eval/deploy path (student.eval.diagnose_merge) AND the training env (dogfight.envs.single_agent_env,
governor-in-the-loop residual RL leg). Single source = train==eval==deploy governor (no drift).

The brake is a STATEFUL override (closure-EMA + arm/disarm latch); its label is bimodal on a fixed obs,
so a memoryless/fv policy cannot distill it (round-2 verdict) — but a wrapper gets the latch state for
free. Inputs (range/az/closure) are deploy-wire computable from the tac24 base channels o[6,7,8,11,22];
GCAS already ships a scripted override so this is deploy-legal. Returns a SIM-space throttle CAP in
[0,1] (1.0 = no cap). Mutates st['clo_s','brk'].
ponytail: closure-EMA latch (2 state keys). If throttle oscillates at the settle boundary, widen the
arm/disarm hysteresis band before adding a phase timer.
"""
from __future__ import annotations
import numpy as np


def throttle_governor(_o, st, pn=None):
    _p = pn or {}
    _az = float(_o[11]) * 180.0
    _clo_mps = float(_o[22]) * 600.0
    _dnm, _dem, _ddm = float(_o[6]) * 15000.0, float(_o[7]) * 15000.0, float(_o[8]) * 8000.0
    _rng_now = float(np.sqrt(_dnm * _dnm + _dem * _dem + _ddm * _ddm))
    _clo_s = 0.7 * float(st.get("clo_s", _clo_mps)) + 0.3 * _clo_mps       # EMA: reject per-frame noise
    st["clo_s"] = _clo_s
    _arm = float(_p.get("brake_arm", 120.0))
    _arm_rng = float(_p.get("arm_rng", 1500.0))  # ★ energy probe: arm during APPROACH (e.g. 2500) to bleed
    _floor = float(_p.get("floor", 0.15))         #   merge energy BEFORE near-on; deeper floor = harder brake.
    _inband = _rng_now < _arm_rng and abs(_az) < 60.0        # defaults (1500/0.15) = frozen law byte-identical
    # ── arm-2: terminal-pocket settle (memoryless, closure-conditional; default OFF = byte-identical) ──
    # 260725 P-A: arm-1 (blow-through) needs clo_s > brake_arm (~100), but the last-2s conversion pocket closes
    # at only +26..52 m/s, so the brake NEVER arms there — the kill-chain blind spot (funnel_probe H4). arm-2 caps
    # throttle to SETTLE the nose in the cone (teacher's 0.28 mechanism -> cone dwell). Reads o[16]=ATA, which is
    # attitude(roll/pitch/heading)+relpos only (NO velocity => deploy-exact; deploy carries real attitude). Closure-
    # gated (overtake-conditional-modifiers-law); energy-only cap (charter addendum B); no latch (round-2: latch->
    # bimodal). pocket_on=0 (default) => _cap stays 1.0 => every return path byte-identical to the frozen law.
    _cap = 1.0
    if float(_p.get("pocket_on", 0.0)) > 0.5:
        _ata = abs(float(_o[16])) * 10.0                     # obs[16]=normalize(ata,-10,10), saturates at 10deg
        if (_rng_now < float(_p.get("pocket_rng", 914.0))
                and _ata < float(_p.get("pocket_ata", 10.0))
                and _clo_s > float(_p.get("pocket_arm", 20.0))):
            _cap = float(_p.get("pocket_cap", 0.30))
    # ── arm-3: CORNER-SPEED in the turning fight (default OFF = byte-identical) ─────────────────────
    # 260726, user hypothesis confirmed by measurement: the champion holds throttle 1.00 in EVERY speed
    # bin, so it fights above corner speed and its turn circle is huge. Measured median turn radius by
    # (altitude x speed): at 2-5 km it is 1988 m above 260 m/s but 438 m below 170 m/s -- speed swings
    # the radius 2.6x, altitude only 1.6x. A wide circle is why the two jets stop crossing (few windows)
    # and it is what reads as "slow, monotone" in the viewer. corner_kcas 340 (~175 m/s) is NOT a new
    # number: student/selfplay/scripted_opponent.py already flies this exact law (R = V^2/(g tan phi)).
    # Reads o[3] = KCAS, a FIRST-CLASS PlaneInfo field on the deploy wire (native_bt.PlaneInfo.KCAS) --
    # not position-differenced, so it clears the 2nd-order obs gate like range/az/closure do.
    # OVERTAKE-CONDITIONAL (project law: brake-v1/lag-v1 both died by slowing down against an extender):
    # this arm is DISARMED whenever the bandit is running -- range beyond `corner_rng_max`, or opening
    # (clo_s below `corner_open`). Out-boxing is a competition penalty, so fleeing must always win the
    # throttle back. corner_on=0 (default) => untouched.
    # ★★ CHANNEL CALIBRATION (260726, measured — do NOT set corner_kcas in textbook knots): decoding
    # obs[3] as (x+1)*300 yields ~HALF the true airspeed. Verified against position-differenced ground
    # speed in three altitude bands: <2km ch221 vs 231 m/s (=442 kt, ratio 2.03); 5-8km ch93 -> 186 kt
    # CAS = 133 m/s TAS vs 129 measured; >8km ch63 -> 104 vs 106 measured. Physics agrees with the ground
    # speed and not the channel: 5.7 g at the channel's 114 m/s would exceed max lift, at 231 m/s it is
    # routine. So corner_kcas is in CHANNEL units ~= 0.5 x true KCAS (160 channel ~= 320 kt true). The
    # first probe used 340 (= 680 kt true) and consequently never fired once -- same trap family the GCAS
    # header calls "KCAS-2x". Keep this arm keyed to the channel; never mix the two scales.
    if float(_p.get("corner_on", 0.0)) > 0.5:
        _kcas = (float(_o[3]) + 1.0) * 300.0                 # obs[3]=normalize(KCAS,0,600); CHANNEL units
        _turning = (abs(_az) > float(_p.get("corner_az", 25.0))
                    and float(_p.get("corner_rng_min", 900.0)) < _rng_now < float(_p.get("corner_rng_max", 4000.0))
                    and _clo_s > float(_p.get("corner_open", -25.0)))   # NOT extending away
        if _turning:
            _target = float(_p.get("corner_kcas", 340.0))
            # proportional bleed: idle when fast, back to mil as it decays to corner (scripted_opponent law)
            _cmd = float(np.clip(0.55 + float(_p.get("corner_k", 0.004)) * (_target - _kcas),
                                 float(_p.get("corner_idle", 0.05)), 1.0))
            _cap = min(_cap, _cmd)
    # ── arm-4: DEFENSIVE-BREAK energy cap (default OFF = byte-identical) ─────────────────────────────
    # 260809. At 760 m with a bandit on the six, EVERY subject we can fly leaves the fight: champion
    # (with and without governor/takeover), two different NNs, and the hand-coded merge-brake teacher --
    # all 12/12 disengaged at 27-29 s with margin -6.6..-6.8%. Identical numbers from a P-controller and
    # a neural net means this is not a learned choice, it is the physics of the break: the defensive
    # response is a hard nose-UP pull, and a hard pull at FULL THROTTLE converts into a climb (measured:
    # 760 -> 5235 m while the bandit stays at ~900 m), and the climb IS the separation that trips the
    # disengage. That also explains why the re-engage arm failed here: it handed control to the teacher,
    # a pilot with the SAME defect. Pull is the fragile channel and stays untouched; throttle is the
    # governor's legal channel, so remove the energy that feeds the zoom and let the break stay in-plane.
    # Deliberately NOT closure-gated (unlike arm-1/3): the whole point is that we ARE opening -- the
    # overtake-conditional law guards against braking while CHASING an extender, and here the bandit is
    # behind us, not fleeing. Disarms as soon as the bandit is no longer in the rear hemisphere.
    if float(_p.get("defbrk_on", 0.0)) > 0.5:
        if (abs(_az) > float(_p.get("defbrk_az", 110.0))
                and _rng_now < float(_p.get("defbrk_rng", 3000.0))):
            _cap = min(_cap, float(_p.get("defbrk_cap", 0.45)))
    if _inband and _clo_s > _arm:              # caught a turning target fast -> about to blow through
        st["brk"] = 1.0
    elif _rng_now > _arm_rng + 100.0 or _clo_s < -15.0:  # target extended out of band / fleeing -> full chase
        st["brk"] = 0.0
    if float(st.get("brk", 0.0)) > 0.5 and _inband:
        return min(_cap, float(np.clip(0.55 - 0.02 * (_clo_mps - 35.0), _floor, 1.0)))
    return _cap                                 # arm-2/3 cap (or 1.0 = no cap when OFF / outside their gates)
