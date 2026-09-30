# -*- coding: utf-8 -*-
"""Autopilot merge-teacher opponent — robust pursuit via the VENDOR flight computer.

Why (workflow w73oysbn2 + docs/reward-structure-research-260614.md): 8 hand-coded
stick controllers all FAILED — two mutual-pursuit stick controllers diverged (range
11-18 km, bearing pinned 87-135°, re-merges 0). Root causes (code-verified):
  1. obs[9] (3D ATA) is UNSIGNED in the horizontal (GeoMathUtil arccos) — it cannot
     resolve left/right, so bank-toward-bearing turned the WRONG way on ~half the
     merges. (Fix: use ΔN/ΔE atan2 for a sign-correct bearing.)
  2. The 6-DOF energy/altitude/turn coupling is not hand-tunable from one stick law:
     max pull mushed to 83 m/s (stall spiral); reduced pull → 40% self-crash.
This provider OFFLOADS flight control to FighterSim.step_autopilot (JSBSim AutoRun
0x03 heading-hold / 0x04 altitude-hold / 0x03 speed-hold), which manages bank/pull/
throttle internally to capture a commanded heading while HOLDING altitude and speed —
exactly the coupling the stick law could not. Commanding heading = bearing-to-learner
every frame gives a clean, sustained, level PURSUIT turn. Precedent: campaign v2k
(rear × autopilot) = win 0.95, crash 0.02 (a stable, beatable target).

Output contract: compute_autopilot(context) -> {"heading_cmd": deg[0,360),
"altitude_cmd": m, "speed_cmd": m/s}. compute_action(context) wraps the same dict in
ActionResult.info["autopilot"] so it works on BOTH the env-native dispatch
(single_agent_env._step_target_aircraft) and the Wine-bridge wire.

Anti-self-crash by construction: altitude_cmd clamped >= 3100 m (>> 300 m kill floor)
and speed_cmd clamped >= 185 m/s (above mush), both inside the documented autopilot
envelope (Alt 3048-8534 m, Speed 183-305 m/s — JSBSimWrapper). Beatable: capped speed
=> wide sustained radius => the learner arrives inside first.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

DEFAULTS = {
    "corner_mps": 210.0,        # base commanded speed — low-in-envelope = WIDE radius = beatable
    "alt_hold_m": 7000.0,       # fallback hold altitude when vertical_track is off
    "heading_offset_deg": 0.0,    # heading = atan2(dE,dN) = NED compass bearing toward the learner.
                                  # W0-confirmed CORRECT: offset 0 banks toward the learner.
    # ── THREAT upgrades (260616): Test-1 showed pure-pursuit + FIXED 7000 m alt could not
    #    kill even a passive learner — it descended and the teacher could not follow it down,
    #    nor cut the corner for a gun solution. These three give it real lethality: ──
    # ── LOITER mode (260616, FIXED 260617): pursue=False makes the teacher IGNORE the learner and fly a
    #    genuine constant-rate CIRCLE at fixed alt/speed — a non-ATTACKING gently-turning target. Use for the
    #    2a robustness probe (does the rear gun-track survive a TURNING target, or is it a straight-target
    #    artifact?). A pursuing teacher turns to FACE/attack → degrades into a neutral fight; the loiter is a
    #    clean gentle curve that ignores the learner. ──
    "pursue": True,             # False = loiter (ignore learner, non-attacking target)
    # loiter (FIXED 260617): the old `own_yaw + offset` SELF-CANCELLED → ONE ~60° transient turn then STRAIGHT +
    # flee. FIX: latch the heading at spawn and ADVANCE it by loiter_rate_deg_per_step each call, own_yaw-INDEPENDENT.
    # ★ RATE RE-CALIBRATED (260617, 2b-audit catch): compute_autopilot is called ONCE PER JSBSim SUB-STEP, i.e.
    #   step_ratio(=6) times per RL-step (single_agent_env.py _advance_simulation_step_ratio loop → _step_target_aircraft),
    #   NOT once per RL-step. So the per-call rate is MULTIPLIED by 6×60=360 to get °/s:  rate °/s = 360 × this.
    #   The old 0.5 → 0.5×360 = 180... no: 0.5°/call × 6 calls/RL-step × 10 RL-step/s = 30 °/s COMMANDED (≈11 G at
    #   210 m/s = UNFLYABLE → the setpoint saturated the airframe; empirically the loiter only ACHIEVED ~1.5°/s, a
    #   wide ~8 km circle). For a flyable, trackable ~5°/s gentle circle (≈1.9 G, ~2.4 km radius): 5/60 ≈ 0.083.
    #   loiter_rate IS now a real DIFFICULTY DIAL (it was saturated/dead at 0.5). MUST VERIFY achieved rate via the
    #   eval --dump-trace (누적회전 monotonic + dist bounded) BEFORE trusting it — the autopilot may still gain-limit.
    "loiter_rate_deg_per_step": 0.083,   # ≈ 5°/s commanded (was 0.5 = 30°/s, saturated). Verify achieved rate empirically.
    "loiter_offset_deg": 60.0,  # [DEPRECATED — unused after the 260617 rate fix; kept for back-compat only]
    "loiter_reverse_period_steps": 0,   # 0 = constant circle; >0 = flip the loiter turn direction every N sub-steps (~60/s) = PERIODIC level jinks
    "loiter_reverse_prob": 0.0,         # >0 = flip with this probability EACH sub-step = UNPREDICTABLE jinks (T2: agent cannot pattern-match)
    "loiter_rate_jitter": 0.0,          # >0 = per-EPISODE uniform jitter on the turn rate = VARIED turn speed (T2)
    # ── EVADE-SCAS (T1+ TRACKING curriculum, 260618): a HARD-turning NON-attacking EVADER the learner must
    #    track. loiter/pursue use vendor heading-hold (latCtrlMode 0x03) which GAIN-LIMITS to ~1.5-3°/s (wide
    #    slow drift, NOT a real turn). This dispatches to FighterSim.step_loiter = SCAS (latCtrlMode 0x02) which
    #    commands the BANK ANGLE DIRECTLY → a sustained turn:  ω = g·tan(bank)/V  (bank 60°@210m/s ≈ 4.6°/s,
    #    70° ≈ 7.4°/s, 75°@185 ≈ 11°/s). The target IGNORES the learner = it EVADES, does NOT attack → no neutral
    #    fight, the learner keeps the rear. bank = the difficulty DIAL (T1 moderate ~55-60, T2 hard ~70-75).
    #    reverse_period>0 flips the bank for jinks/reversals (T2 variety); 0 = a constant hard circle (T1).
    #    ★ VERIFY the ACHIEVED rate via eval --dump-trace (누적회전 monotonic↑, dist bounded, suicide 0) BEFORE
    #    trusting it — the vendor SCAS may still G-limit the commanded bank. This is G0, the curriculum long pole.
    "evade_scas": False,                  # True = SCAS hard-bank evader (FlightPathAngle turn; overrides pursue/loiter)
    "evade_bank_deg": 60.0,               # commanded bank angle = the turn-rate difficulty dial
    "evade_gamma_deg": 0.0,               # FlightPathAngle hold: 0 = level turn; negative = descending (T3 energy)
    "evade_speed_mps": None,              # None = spawn speed; LOWER raises turn rate (omega = g*tan(bank)/V)
    "evade_reverse_period_steps": 0,      # 0 = constant circle; >0 = flip bank every N compute-calls (~60/s) = jinks
    "evade_bank_jitter_deg": 0.0,         # per-step random bank jitter (T2 unpredictability)
    # ── REACTIVE DEFENDER (260718, STEP-7 gate gauge): a threat-REACTIVE break-turn defender. Reads its
    #    OWN obs, and when the LEARNER gains its rear hemisphere + closes, commands a max-G defensive BREAK
    #    toward the threat (deny the gun solution / force an overshoot) with a defensive-SCISSORS reversal
    #    when the learner crosses the tail. Otherwise jinks/extends to rebuild energy. This is the missing
    #    "does the champion get a CLEAN KILL vs something that actively DEFENDS?" gauge — the pursuers only
    #    measure P(kill|the opponent presses); a pure evader ignores the learner; this one FIGHTS BACK
    #    defensively. bank-direct via step_scas_turn (omega = g*tan(bank)/V). Never self-crashes (2-tier floor).
    "defend_reactive": False,             # True = reactive break-turn defender (overrides pursue/evade/loiter)
    "defender_strength": 0.7,             # [0,1] beatability dial: scales break bank + jink hardness + corner tightness
    "break_range_m": 2600.0,              # learner nearer than this (and closing, rear) => BREAK
    "break_cone_deg": 55.0,               # learner off the nose by more than this (toward the rear) => threatened
    "defender_alt_floor_m": 1500.0,       # hard wings-level climb below this (anti self-crash)
    "defender_alt_soft_m": 3000.0,        # level off + gentle climb below this when descending
    "strength": 1.0,            # ★ master beatability dial [0,1]: scales lead + closure (0=soft pure-pursuit)
    "lead_gain": 3.0,           # PN-like lead on the LOS rate: heading = bearing + gain*LOS_rate (cut the corner)
    "lead_max_deg": 30.0,       # clamp on the lead angle
    "vertical_track": 1.0,      # 1 = FOLLOW the learner's altitude (clamped); 0 = fixed alt_hold_m
    "closure_boost_mps": 45.0,  # extra speed beyond closure_range — close the gap to a kill
    "closure_range_m": 1500.0,  # apply closure boost when distance exceeds this
    "near_range_m": 1200.0,     # inside this: optional lag bias (beatability)
    "lag_deg": 0.0,             # optional near-range under-turn (extra beatability); 0 = off
    "alt_clamp": [3100.0, 8500.0],   # stay inside the autopilot envelope (alt-hold floor = anti-self-crash)
    "speed_clamp": [185.0, 300.0],
    "speed_jitter": 15.0,       # per-episode anti-cheese distribution of competent turners
    "lag_jitter": 3.0,
    "seed": 0,
}


def _merge(spec: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(spec or {})
    cfg["alt_clamp"] = list(cfg["alt_clamp"])
    cfg["speed_clamp"] = list(cfg["speed_clamp"])
    return cfg


class AutopilotTeacherProvider(ActionProvider):
    """Pursues the learner via the vendor autopilot (heading-hold toward learner)."""

    def __init__(self, spec: dict | None = None):
        self.spec = _merge(spec)
        seed = int(self.spec.get("seed", 0))
        # ★ 260809 REPRODUCIBILITY FIX: this was np.random.default_rng(abs(hash((seed, "autopilot_teacher"))) % 2**32).
        # Python randomizes hash() of str/tuple-containing-str per PROCESS unless PYTHONHASHSEED is set, so the
        # opponent's per-episode jitter drew from a DIFFERENT stream on every run -- identical bundle + identical
        # env seed did not reproduce. Measured on band_g1 vs jittered_teacher, same bundle, same seed 4321:
        # +73.6%/3 kills on one run vs +27.4%/1 kill on another (46 pt swing). Every jteacher-cell number this
        # project ever recorded carries that uncontrolled variance. zlib.crc32 over stable bytes is deterministic
        # across processes and interpreters; the per-episode variety (the point of the jitter) is unchanged.
        self._rng = np.random.default_rng(zlib.crc32(b"autopilot_teacher:%d" % seed))
        self._corner = float(self.spec["corner_mps"])
        self._lag = float(self.spec["lag_deg"])
        self._strength = float(np.clip(self.spec.get("strength", 1.0), 0.0, 1.0))
        self._prev_bearing = None   # for the PN-like LOS-rate lead
        self._loiter_heading = None  # for the constant-rate loiter turn
        self._loiter_sign = 1.0      # ★ RANDOM turn direction per episode (set in reset)
        self._loiter_step = 0        # loiter compute-call counter (for reversals)
        self._loiter_rate = float(self.spec.get("loiter_rate_deg_per_step", 0.5))  # per-episode rate (jittered in reset)
        self._evade_sign = 1.0       # SCAS evade turn direction (random per episode, set in reset)
        self._evade_step = 0         # SCAS evade compute-call counter (for reversals)
        self._evade_gamma_sign = 1.0 # ★ 260718 SCAS evade gamma (climb/dive) sign — flips for vertical yo-yo
        self._prev_bank = 0.0        # SCAS-pursuit per-SUB-STEP slew memory (anti roll-slam; reset per episode)
        self._defend_break_sign = 0.0  # ★ 260718 reactive-defender break/scissors direction latch (reset per episode)
        self._prev_los_sign = None     # ★ 260718 previous LOS-az sign, for scissors overshoot detection

    def reset(self, context: ActionContext | None = None) -> None:
        sj = float(self.spec.get("speed_jitter", 0.0))
        lj = float(self.spec.get("lag_jitter", 0.0))
        self._corner = float(self.spec["corner_mps"] + self._rng.uniform(-1, 1) * sj)
        self._lag = float(self.spec["lag_deg"] + self._rng.uniform(-1, 1) * lj)
        self._prev_bearing = None
        self._loiter_heading = None
        self._loiter_sign = float(self._rng.choice([-1.0, 1.0]))   # ★ random L/R curve per episode
        self._loiter_step = 0
        _base = float(self.spec.get("loiter_rate_deg_per_step", 0.5))
        _rj = float(self.spec.get("loiter_rate_jitter", 0.0))
        self._loiter_rate = _base + (float(self._rng.uniform(-1.0, 1.0)) * _rj if _rj > 0.0 else 0.0)  # VARIED speed per episode
        self._evade_sign = float(self._rng.choice([-1.0, 1.0]))    # SCAS evade L/R per episode
        self._evade_step = 0
        self._evade_gamma_sign = float(self._rng.choice([-1.0, 1.0]))   # ★ 260718 random climb/dive start per episode
        self._prev_bank = 0.0
        self._defend_break_sign = 0.0   # ★ 260718 reactive-defender: no committed break direction at spawn
        self._prev_los_sign = None

    def compute_autopilot(self, context: ActionContext) -> dict:
        """The 3 scalars consumed by FighterSim.step_autopilot.

        altitude_cmd is NED-DOWN m (step_autopilot passes -altitude_cmd*M2FT to AutoRun
        whose alt_cmd is +feet-up), so X m UP is commanded as -X.
        """
        sc = self.spec["speed_clamp"]
        ac = self.spec["alt_clamp"]
        alt_fallback_ned = -float(np.clip(self.spec["alt_hold_m"], ac[0], ac[1]))
        obs = context.observation
        if obs is None:
            return {"heading_cmd": 0.0, "altitude_cmd": alt_fallback_ned,
                    "speed_cmd": float(np.clip(self._corner, sc[0], sc[1]))}
        o = np.asarray(obs, dtype=np.float32).ravel()
        if o.shape[-1] > 24 and o.shape[-1] != 29:   # ★ 260807 fv/h stack -> base frame (see _compute_pursuit_scas)
            o = o[:24]

        # ── LOITER (pursue=False): IGNORE the learner — fly a constant-rate turn at fixed
        #    alt/speed = a non-ATTACKING moving target. Keeps the learner's REAR advantage
        #    (a pursuing teacher turns to FACE/attack → degrades into a neutral turning fight). ──
        if not bool(self.spec.get("pursue", True)):
            # ★ TRUE constant-rate circle: latch the heading reference ONCE at spawn, then ADVANCE it by a
            #   fixed rate every step — DECOUPLED from own_yaw (re-reading own_yaw self-cancels the error →
            #   one transient turn then flee, the 260617 bug). An own_yaw-independent advancing setpoint makes
            #   the heading-hold hold a genuine steady gentle turn. _loiter_sign = random L/R per episode.
            if self._loiter_heading is None:
                self._loiter_heading = (float(o[2]) + 1.0) * 180.0   # seed from spawn heading (obs[2]∈[-1,1]→[0,360)°)
            # REVERSALS: PERIODIC (reverse_period) OR RANDOM (reverse_prob = UNPREDICTABLE jinks) = LEVEL jinks (no descent).
            rev = int(self.spec.get("loiter_reverse_period_steps", 0))
            rp = float(self.spec.get("loiter_reverse_prob", 0.0))
            if (rev > 0 and self._loiter_step > 0 and (self._loiter_step % rev) == 0) or \
               (rp > 0.0 and float(self._rng.random()) < rp):
                self._loiter_sign = -self._loiter_sign
            self._loiter_step += 1
            rate = self._loiter_sign * self._loiter_rate   # per-episode (jittered) rate
            self._loiter_heading = (self._loiter_heading + rate) % 360.0   # ~5°/s at the bridge's 0.1 s/step
            return {"heading_cmd": float(self._loiter_heading),
                    "altitude_cmd": alt_fallback_ned,   # fixed hold (non-suicide)
                    "speed_cmd": float(np.clip(self._corner, sc[0], sc[1]))}

        # ΔN/ΔE/ΔD to the learner (same ±15000/±8000 m scales used at build time).
        dn, de, dd = float(o[6]) * 15000.0, float(o[7]) * 15000.0, float(o[8]) * 8000.0
        dist = float((dn * dn + de * de + dd * dd) ** 0.5)
        # atan2(ΔE, ΔN) = NED compass bearing toward the learner (sign-correct, unlike obs[9]).
        bearing = math.degrees(math.atan2(de, dn))

        # ── LEAD pursuit (PN-like on the LOS rate): turn FASTER than the line-of-sight
        #    rotates so the nose cuts inside toward an intercept, instead of pure-pursuit
        #    lag that never closes the gun angle on a co-speed target. ──
        lead = 0.0
        if self._prev_bearing is not None and self._strength > 0.0:
            rate = ((bearing - self._prev_bearing + 180.0) % 360.0) - 180.0   # signed deg/step
            lead = self._strength * float(self.spec["lead_gain"]) * rate
            lm = float(self.spec["lead_max_deg"])
            lead = max(-lm, min(lm, lead))
        self._prev_bearing = bearing
        heading = (bearing + lead + float(self.spec["heading_offset_deg"])) % 360.0

        # Optional near-range under-turn (lag) for extra beatability; 0 by default.
        if self._lag != 0.0 and dist < float(self.spec["near_range_m"]):
            yaw = (float(o[2]) + 1.0) * 180.0
            err = ((heading - yaw + 180.0) % 360.0) - 180.0
            heading = (yaw + err - math.copysign(self._lag, err)) % 360.0

        # ── VERTICAL tracking: FOLLOW the learner down/up (clamped to the non-suicide
        #    floor). Test 1's miss was a FIXED 7000 m alt vs a descending learner. ──
        if float(self.spec.get("vertical_track", 1.0)) >= 0.5:
            own_alt = float(o[4]) * 15000.0          # teacher altitude (up, m)
            learner_alt = own_alt - dd               # dd = own_alt_up - learner_alt_up
            alt_ned = -float(np.clip(learner_alt, ac[0], ac[1]))
        else:
            alt_ned = alt_fallback_ned

        # ── CLOSURE speed: speed up to close the gap when far; base speed near the WEZ. ──
        speed = self._corner
        if dist > float(self.spec["closure_range_m"]):
            speed = self._corner + self._strength * float(self.spec["closure_boost_mps"])

        return {
            "heading_cmd": float(heading),
            "altitude_cmd": alt_ned,   # NED-Down (negative = up)
            "speed_cmd": float(np.clip(speed, sc[0], sc[1])),
        }

    def _compute_evade(self, context: ActionContext) -> dict:
        """SCAS hard-bank evasion → consumed by the env as step_loiter(isLoitering, bank, pitch).

        The target IGNORES the learner and just turns hard at the commanded bank (a sustained
        high-G break/circle the gain-limited heading-hold cannot produce) = a non-attacking
        EVADER the learner must track. reverse_period>0 flips the bank for jinks (T2+); 0 = a
        constant hard circle (T1). bank carries the per-episode L/R sign + optional jitter.
        NOTE: called once per JSBSim sub-step (≈60/s), so reverse_period is in sub-steps.
        """
        bank = float(self.spec.get("evade_bank_deg", 60.0))
        rev = int(self.spec.get("evade_reverse_period_steps", 0))
        if rev > 0 and self._evade_step > 0 and (self._evade_step % rev) == 0:
            self._evade_sign = -self._evade_sign
        jit = float(self.spec.get("evade_bank_jitter_deg", 0.0))
        if jit > 0.0:
            bank = bank + float(self._rng.uniform(-1.0, 1.0)) * jit
        # ★ 260718 GAMMA-REVERSE = vertical yo-yo (climb/dive oscillation) — mirrors the bank reverse
        #   above. Flips the flight-path-angle sign every N sub-steps so a fixed +gamma becomes a
        #   climb→dive→climb pattern. The alt-floor override below still guarantees no self-crash.
        grev = int(self.spec.get("evade_gamma_reverse_period_steps", 0))
        if grev > 0 and self._evade_step > 0 and (self._evade_step % grev) == 0:
            self._evade_gamma_sign = -self._evade_gamma_sign
        self._evade_step += 1
        bank = self._evade_sign * bank
        gamma = self._evade_gamma_sign * float(self.spec.get("evade_gamma_deg", 0.0))
        speed = self.spec.get("evade_speed_mps", None)
        # ★ ALT-FLOOR (260630): mirror the threat's PROVEN 2-tier floor (cut self-crash 18.7%->0%, lines 339-355) into the
        #   evader. The un-floored _compute_evade self-crashed (the "death-spiral that killed _compute_evade" the threat
        #   docstring cites). USER REQ: the bot must NEVER crash (a self-crash = a free reset + no skill learned + metric
        #   contamination). Decode own_alt obs-mode-aware (tac24 obs[4]=norm(ALT,0,15000); minimal obs[5]=norm(ALT,200,10000)).
        obs = context.observation
        if obs is not None:
            o = np.asarray(obs, dtype=np.float32).ravel()
            own_alt = (float(o[4]) * 7500.0 + 7500.0) if o.shape[-1] <= 24 else (float(o[5]) * 4900.0 + 5100.0)
            hard_floor = float(self.spec.get("evade_alt_floor_m", 1500.0))
            soft_floor = float(self.spec.get("evade_alt_soft_m", 3000.0))
            if own_alt < hard_floor:
                bank = 0.0; gamma = 30.0          # WINGS-LEVEL hard climb (survival > the turn)
            elif own_alt < soft_floor and gamma < 0.0:
                gamma = 5.0                        # descending toward the floor -> level off + gentle climb
        return {"bank": bank, "gamma": gamma, "speed": speed}

    def _compute_pursuit_scas(self, context: ActionContext) -> dict:
        """Hand-scripted LEAD-pursuit THREAT (kill-gate ③). Reads the OPPONENT's own teacher-centric
        minimal_v1 obs (target = the learner) and emits {bank,gamma,speed} for step_scas_turn (bank-DIRECT:
        omega=g*tan(bank)/V, NOT the gain-limited heading-hold that gave dmg_recv~0). Beats both prior
        failure modes BY CONSTRUCTION: bank-direct turn-rate + ENERGY-GOVERNED (sustained-g cap + speed bleed
        + KCAS floor) so it never the fixed-bank/speed-hold death-spiral that killed _compute_evade.
        kappa in [0,1] (spec['kappa']) scales lead/g/burst/closure monotonically; the PD+slew+floor STABILITY
        core is kappa-independent. Called ~60Hz (per JSBSim sub-step) so slew is in deg/sub-step.
        ★KCAS_FLOOR=150 (NOT the design draft's 300): obs[4]*160+200 runs 133-360 (median 184) in real fights,
        so 300 would force-unload 76%% of steps = crippled. 150 unloads only at genuine near-stall (p10).
        """
        sc = self.spec.get("speed_clamp", [185.0, 320.0])
        obs = context.observation
        if obs is None:
            return {"bank": 0.0, "gamma": 0.0, "speed": float(np.clip(self._corner, sc[0], sc[1]))}
        o = np.asarray(obs, dtype=np.float32).ravel()
        # ★ obs-mode-AWARE decode (260628): threaten in BOTH minimal_v1 (29-dim, FDM-sourced R/Q, training-only)
        # AND the DEPLOYABLE tactical23/24 (<=24-dim, kinematic 'line' features at obs[17-22]) so the SAME PD
        # stability core can be the raw_stick-deployable THREAT anchor that punishes disengage. Branch on obs
        # length; indices/scales verified vs observation.py (tac16 base 394-406 + tac23 kinematic 111-116).
        # ★ 260807 FV-WIDTH FIX (band_g1 pre-launch adversarial review, BLOCKER): under tactical24_fv/_h the
        #   env hands EVERY opponent provider the HISTORY-AUGMENTED vector (single_agent_env:1760-1765; fv =
        #   24 base + 48 tail = 72). The old `<=24` branch sent 72 into the minimal_v1 (29-dim) decode, which
        #   reads KCAS from the ALTITUDE channel and own-altitude from the HEALTH channel — so every scasevade
        #   opponent flew "near-stall" (bank hard-capped 35 deg) with dead altitude floors whenever the learner
        #   used an fv/h obs mode. The base frame is always the FIRST 24 channels of every tac24-family stack
        #   (observation.py _augment_fv_tail_params: concatenate([base]+tail)), so slice it off. minimal_v1 is
        #   exactly 29-dim and keeps its branch; genuine <=24 vectors are unchanged.
        if o.shape[-1] > 24 and o.shape[-1] != 29:   # tac24_fv (72) / tac24_h (28) stacks -> base frame
            o = o[:24]
        if o.shape[-1] <= 24:              # tactical23 / tactical24 (the deployable obs)
            los_az = float(o[11]) * 180.0  # LOS_az (same index/scale as minimal)
            los_el = float(o[12]) * 90.0
            az_rate = float(o[20]) * 40.0  # LOS-rate az (tac23 obs[20]; minimal had obs[18])
            el_rate = float(o[21]) * 40.0
            closure = float(o[22]) * 600.0
            rvx = float(o[17]) * 600.0     # rel_vel_body (tac23 obs[17-19] @600; minimal obs[15-17] @300)
            rvy = float(o[18]) * 600.0
            rvz = float(o[19]) * 600.0
            dn, de, dd = float(o[6]) * 15000.0, float(o[7]) * 15000.0, float(o[8]) * 8000.0
            kcas = float(o[3]) * 300.0 + 300.0   # KCAS normalize(0,600); minimal was obs[4] normalize(40,360)
            own_alt = float(o[4]) * 7500.0 + 7500.0   # threat's OWN altitude (tac24 obs[4]=normalize(ALT,0,15000))
        else:                              # minimal_v1 (FDM-sourced R/Q — training-only, non-deployable)
            los_az = float(o[11]) * 180.0  # + = learner to MY right
            los_el = float(o[12]) * 90.0   # + = learner ABOVE
            az_rate = float(o[18]) * 40.0  # deg/s, + = learner crossing right
            el_rate = float(o[20]) * 40.0
            closure = float(o[22]) * 600.0 # + = closing on learner
            rvx = float(o[15]) * 300.0
            rvy = float(o[16]) * 300.0
            rvz = float(o[17]) * 300.0
            dn, de, dd = float(o[8]) * 25000.0, float(o[9]) * 25000.0, float(o[10]) * 8000.0
            kcas = float(o[4]) * 160.0 + 200.0
            own_alt = float(o[5]) * 4900.0 + 5100.0   # threat's OWN altitude (minimal obs[5]=normalize(ALT,200,10000))
        dist = math.sqrt(dn * dn + de * de + dd * dd)
        cross_deg = math.degrees(math.atan2(math.hypot(rvy, rvz), abs(rvx) + 1e-6))
        kappa = float(np.clip(self.spec.get("kappa", 0.8), 0.0, 1.0))
        # kappa-independent stability core (verified PD from the conversion controller)
        KP_AZ, KD_AZ, KP_EL, KD_EL = 1.7, 0.40, 1.1, 0.30
        SLEW_DEG = 3.0          # max |d(bank)| per SUB-STEP (~18 deg/RL-step) — anti roll-slam
        KCAS_FLOOR = 150.0      # ★ genuine near-stall (obs[4] median 184); below -> unload to rebuild E
        WEZ_MAX = 914.4
        # kappa-scaled aggression
        LEAD_GAIN = 0.18 + 0.22 * kappa
        LEAD_MAX = 8.0 + 14.0 * kappa
        N_SUST = 1.8 + 0.7 * kappa
        BANK_BURST = 66.0 + 12.0 * kappa
        V_CLOSE = 250.0 + 50.0 * kappa
        V_TURN = 200.0 - 10.0 * kappa
        GAMMA_MAX = 22.0 + 8.0 * kappa
        # 1) LATERAL lead-pursuit (nose AHEAD of the learner) + damped PD
        lead = float(np.clip(LEAD_GAIN * az_rate, -LEAD_MAX, LEAD_MAX))
        aim_az = los_az + lead
        bank_raw = KP_AZ * aim_az - KD_AZ * az_rate
        # 2) ENERGY GOVERNOR: sustained-g cap default; energy-gated transient burst when lined up
        bank_sust_cap = math.degrees(math.acos(min(0.999, 1.0 / N_SUST)))
        lined_up = (abs(aim_az) < 25.0) and (dist < 2500.0)
        energy_ok = (kcas > KCAS_FLOOR + 20.0)
        bank_cap = BANK_BURST if (lined_up and energy_ok) else min(BANK_BURST, bank_sust_cap)
        if kcas < KCAS_FLOOR:
            bank_cap = min(bank_cap, 35.0)
        bank_des = float(np.clip(bank_raw, -bank_cap, bank_cap))
        bank = self._prev_bank + float(np.clip(bank_des - self._prev_bank, -SLEW_DEG, SLEW_DEG))
        bank = float(np.clip(bank, -80.0, 80.0))
        self._prev_bank = bank
        # 3) VERTICAL lead-pursuit
        gamma = float(np.clip(KP_EL * los_el + KD_EL * el_rate, -GAMMA_MAX, GAMMA_MAX))
        gamma = float(np.clip(gamma, -30.0, 30.0))
        # 4) SPEED: bleed in the hard turn; run down when aligned/far; settle in the shell
        if cross_deg > 45.0 or abs(aim_az) > 30.0:
            speed = V_TURN
        elif cross_deg > 22.0:
            speed = 210.0
        elif dist > 1400.0:
            speed = V_CLOSE
        elif dist > WEZ_MAX:
            speed = 0.5 * (V_CLOSE + 205.0)
        else:
            speed = 205.0
        if dist < 1300.0 and closure > 70.0:
            speed = min(speed, 205.0)
        if kcas < KCAS_FLOOR:
            speed = max(speed, V_CLOSE)
        speed = float(np.clip(speed, sc[0], sc[1]))
        # ★ THREAT ALT-FLOOR (260628, ROBUST 2-tier): the single 900m trigger STILL self-crashed 18.7% (v3 data)
        # because the kappa dive (gamma to -30deg ~ -125 m/s sink) punches through the 300m deck before a 900m climb
        # arrests it, and the 40deg bank bled the climb's vertical component. Mirror the learner's GCAS: (1) level off
        # EARLY when descending below the soft floor, (2) a hard WINGS-LEVEL climb below the hard floor. A self-crashing
        # bot gives free wins + no clean loss-pressure = not a real opponent (the user's requirement: the bot must
        # NEVER crash). soft=3000 keeps it above the merge-fight floor; it pursues in bank but does NOT dive to the deck.
        hard_floor = float(self.spec.get("threat_alt_floor_m", 1500.0))
        soft_floor = float(self.spec.get("threat_alt_soft_m", 3000.0))
        if own_alt < hard_floor:
            gamma = 30.0                                  # hard climb
            bank = 0.0                                    # WINGS-LEVEL -> full lift goes vertical (survival > pursuit)
            speed = max(speed, V_CLOSE)
        elif own_alt < soft_floor and gamma < 0.0:
            gamma = 5.0                                   # descending toward the floor -> level off + gentle climb, keep the pursuit bank
            speed = max(speed, V_CLOSE)
        self._prev_bank = bank
        speed = float(np.clip(speed, sc[0], sc[1]))       # re-clip (the floor may have raised speed past the band)
        return {"bank": bank, "gamma": gamma, "speed": speed}

    def _compute_reactive_defender(self, context: ActionContext) -> dict:
        """Threat-REACTIVE break-turn DEFENDER (STEP-7 gate gauge) -> {bank,gamma,speed} for step_scas_turn.

        Reads its OWN (defender-centric) obs where the 'target' IS the learner. When the learner gains the
        rear hemisphere and closes, it commands a max-G defensive BREAK toward the threat (turning INTO the
        bandit raises aspect + LOS-rate = defeats the tracking/gun solution and forces an overshoot); when
        the learner crosses the tail (los_az sign flips) it REVERSES the break = a defensive SCISSORS that
        denies a stable saddle. Not currently threatened but still in the fight => a moderate jink in the
        last break direction (no straight-line flee = keeps the gauge engaged). Shaken/far => a gentle
        energy-rebuilding turn back toward the learner. Never self-crashes (the same 2-tier alt floor the
        threat/evader use). The champion must OUT-fly this to land a clean gun kill => the 'clean kill vs a
        DEFENDER > 0' step-7 criterion, which the pure pursuers/evaders cannot measure.
        Called ~60 Hz (per JSBSim sub-step), so the slew is deg/sub-step.
        """
        sc = self.spec.get("speed_clamp", [185.0, 320.0])
        obs = context.observation
        if obs is None:
            return {"bank": 0.0, "gamma": 0.0, "speed": float(np.clip(self._corner, sc[0], sc[1]))}
        o = np.asarray(obs, dtype=np.float32).ravel()
        if o.shape[-1] > 24 and o.shape[-1] != 29:   # ★ 260807 fv/h stack -> base frame (see _compute_pursuit_scas)
            o = o[:24]
        # obs-mode-aware decode (identical mapping to _compute_pursuit_scas: tac23/24 <=24-dim vs minimal_v1).
        if o.shape[-1] <= 24:
            los_az = float(o[11]) * 180.0
            closure = float(o[22]) * 600.0     # + = range decreasing (learner gaining when it is behind)
            dn, de, dd = float(o[6]) * 15000.0, float(o[7]) * 15000.0, float(o[8]) * 8000.0
            own_alt = float(o[4]) * 7500.0 + 7500.0
        else:
            los_az = float(o[11]) * 180.0
            closure = float(o[22]) * 600.0
            dn, de, dd = float(o[8]) * 25000.0, float(o[9]) * 25000.0, float(o[10]) * 8000.0
            own_alt = float(o[5]) * 4900.0 + 5100.0
        dist = math.sqrt(dn * dn + de * de + dd * dd)
        kappa = float(np.clip(self.spec.get("defender_strength", 0.7), 0.0, 1.0))
        BREAK_BANK = 68.0 + 12.0 * kappa                 # 68..80 deg — the harder the more evasive
        BREAK_RANGE = float(self.spec.get("break_range_m", 2600.0))
        BREAK_CONE = float(self.spec.get("break_cone_deg", 55.0))
        SLEW_DEG = 3.5
        V_CORNER = 195.0 - 5.0 * kappa                   # slower = tighter radius = harder to track

        # THREAT: the learner is off my nose toward the rear hemisphere, near, and the range is closing.
        rear = abs(los_az) > BREAK_CONE
        near = dist < BREAK_RANGE
        closing = closure > -20.0
        threatened = near and closing and (rear or dist < 1200.0)

        # ── DEFENSIVE SCISSORS: break TOWARD the threat; when the learner crosses to the other side
        #    (los_az sign flips, away from the ±180 wrap) REVERSE toward the new side so the attacker
        #    overshoots again — denies a stable tracking saddle. ──
        los_sign = 1.0 if los_az >= 0.0 else -1.0
        if threatened:
            if self._prev_los_sign is not None and los_sign != self._prev_los_sign and abs(los_az) < 150.0:
                self._defend_break_sign = los_sign        # follow the threat across the tail (scissors reversal)
            elif self._defend_break_sign == 0.0:
                self._defend_break_sign = los_sign         # first break: turn into the bandit
        self._prev_los_sign = los_sign

        if threatened:
            bank_des = self._defend_break_sign * BREAK_BANK
            gamma = -3.0                                   # slight nose-low to hold corner speed in the break
            speed = V_CORNER
        elif dist < 3500.0:
            # still in the fight but not on my six: keep a MODERATE jink in the last break direction —
            # deny a clean re-entry without fleeing straight (the runner/no-convert artifact).
            s = self._defend_break_sign if self._defend_break_sign != 0.0 else los_sign
            bank_des = s * (40.0 + 15.0 * kappa)
            gamma = 0.0
            speed = V_CORNER + 15.0
        else:
            # shaken / far: a gentle level turn back toward the learner to keep the gauge engaged + rebuild E.
            bank_des = float(np.clip(los_az / 3.0, -35.0, 35.0))
            gamma = 2.0
            speed = 235.0

        bank = self._prev_bank + float(np.clip(bank_des - self._prev_bank, -SLEW_DEG, SLEW_DEG))
        bank = float(np.clip(bank, -82.0, 82.0))
        gamma = float(np.clip(gamma, -20.0, 25.0))
        speed = float(np.clip(speed, sc[0], sc[1]))
        # ★ ALT-FLOOR (never self-crash): same 2-tier guard as the threat/evader (a self-crashing defender
        #   = a free win + no clean-kill pressure = not a real gauge).
        hard_floor = float(self.spec.get("defender_alt_floor_m", 1500.0))
        soft_floor = float(self.spec.get("defender_alt_soft_m", 3000.0))
        if own_alt < hard_floor:
            bank = 0.0; gamma = 30.0; speed = max(speed, 235.0)     # wings-level hard climb (survival > the break)
        elif own_alt < soft_floor and gamma < 0.0:
            gamma = 5.0                                             # descending toward the floor -> level off
        self._prev_bank = bank
        speed = float(np.clip(speed, sc[0], sc[1]))
        return {"bank": bank, "gamma": gamma, "speed": speed}

    def compute_action(self, context: ActionContext) -> ActionResult:
        # REACTIVE DEFENDER (STEP-7 gate gauge): break-turn away from the learner's rear threat.
        if bool(self.spec.get("defend_reactive", False)):
            return ActionResult(
                action=np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                source="autopilot",
                info={"scas_turn": self._compute_reactive_defender(context),
                      "opponent_id": "reactive_defender:break"},
            )
        # SCAS PURSUIT (kill-gate ③ threat): bank-direct LEAD-pursuit of the learner -> info["scas_turn"].
        if bool(self.spec.get("pursue_scas", False)):
            return ActionResult(
                action=np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                source="autopilot",
                info={"scas_turn": self._compute_pursuit_scas(context),
                      "opponent_id": "autopilot:pursue_scas"},
            )
        # SCAS evade (T1+ tracking curriculum): the env reads info["scas_turn"] -> step_scas_turn.
        if bool(self.spec.get("evade_scas", False)):
            return ActionResult(
                action=np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
                source="autopilot",
                info={"scas_turn": self._compute_evade(context),
                      "opponent_id": "autopilot:evade_scas"},
            )
        # Dummy stick action; the env reads info["autopilot"] and calls step_autopilot.
        return ActionResult(
            action=np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32),
            source="autopilot",
            info={"autopilot": self.compute_autopilot(context),
                  "opponent_id": "autopilot:merge_teacher"},
        )

    def close(self) -> None:
        return None


__all__ = ["AutopilotTeacherProvider", "DEFAULTS"]
