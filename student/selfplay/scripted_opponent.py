# -*- coding: utf-8 -*-
"""Scripted MERGE-TEACHER opponent (the L1 lever of ACE-LADDER).

Why this exists (docs/reward-structure-research-260614.md):
  commit-risk is a SKILL-CONDITIONED learned-value gap V(commit)≈-100·P(loss|skill).
  The only term multiplied by the 100-point terminal coefficient is P(loss|commit),
  and ONLY opponent behavior moves it — no reward shaping can. A merge teacher must:
    (i)  CONTEST the merge — turn INTO it (so EXTENDING gets the learner gunned →
         extend earns real damage → terminal loss → EV(extend) falls for a real reason),
    (ii) be BEATABLE at close range — wider turn radius / capped g (so a learner that
         commits and pulls arrives inside first → V(commit) becomes genuinely positive),
    (iii) NEVER self-crash — a hard altitude floor + g cap (the BT/loiter failure was
         the spiral dive; a self-crashing target gifts safe draws and teaches nothing).

Control law: bank-toward-bearing pursuit (LEAD far to turn in hard, LAG near to stay
beatable) + capped sustained pull (g limit = beatability) + HARD altitude-floor override
(anti-self-crash). Acts from the OPPONENT's OWN observation vector — in the Wine bridge
the target provider receives observation only (ownship_state is None), so geometry is
decoded from the tactical16/17 obs (teacher = ownship in its own obs):
  obs[0]=roll/180  obs[3]=KCAS/600  obs[4]=ALT/15000
  obs[6,7,8]=Δn/15000, Δe/15000, Δd/8000   obs[11]=LOS_az/180 (SIGNED body-frame: >0 ⇒ right)
  obs[12]=LOS_el/90 (>0 ⇒ learner above). NOTE: obs[9]=ATA is UNSIGNED (arccos∈[0,180]) —
  do NOT use it for the lateral bank; LOS_az (obs[11]) is the correct signed bearing.
Output is in SIM format: [roll,pitch,rudder] ∈ [-1,1], throttle ∈ [0,1] (bridge contract).

Sign note: roll_sign/pitch_sign let the W0 pre-flight flip the turn direction in ONE
config line if the rollout shows the teacher turning AWAY instead of in — verifying the
turn direction is exactly what student/eval/w0_preflight_teacher.py is for.
"""
from __future__ import annotations

import zlib

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult

DEFAULTS = {
    # Beatability / aggression (L1 = most beatable; ladder raises these). Must be
    # aggressive ENOUGH to re-engage after the head-on pass (a too-gentle turn just
    # flies apart — the W0 v2 failure: range blew out to ~10 km), beatable via the
    # LAG pursuit + a bank below the learner's max, not via an un-re-engaging turn.
    "turn_gain": 0.72,          # ⇒ max bank ~54° (a real fighter turn that re-merges)
    "g_pull_cap": 0.45,         # ★ SUSTAINED-turn pull (W0 telemetry: max pull bled 280→83 m/s = mush)
    "max_bank_deg": 75.0,       # phi_max = turn_gain * max_bank_deg
    "pull_clip": 0.55,          # absolute pull ceiling — keep energy, don't bleed into a stall-spiral
    # Lead / lag pursuit
    "lead_range_m": 1500.0,     # beyond: pure/lead pursuit — turn IN HARD to contest / re-merge
    "near_range_m": 700.0,      # inside: switch to LAG (fall behind the solution = beatable)
    "lag_deg": 8.0,             # lag aim offset behind the learner inside near_range_m
    # Gains
    "k_bank": 2.0,              # bearing(deg) → desired-bank(deg) (saturates bank fast on a big error)
    "k_roll": 0.045,            # bank-error(deg) → aileron [-1,1]
    "k_pitch_el": 0.010,        # LOS elevation(deg) → elevator
    "base_pull": 0.30,          # baseline turn pull (sustained, holds energy)
    "roll_sign": 1.0,           # flip if W0 shows it turning away
    "pitch_sign": 1.0,          # FighterSim: -1=aft(pull/UP); pull_cmd bakes the sign in
    # Energy: HOLD CORNER SPEED. Turn radius R=V²/(g·tanφ): at the ~543 KCAS merge a
    # 54° bank is a ~5 km radius / ~58 s reversal (it can NEVER re-engage — W0 v3). A
    # fighter BLEEDS to corner (~340 KCAS) where the turn is tight (~1.1 km radius).
    # Throttle is proportional: idle when fast (decelerate to corner), mil when slow.
    "corner_kcas": 340.0,       # TARGET corner speed to hold (tight-turn regime)
    "cruise_throttle": 0.55,    # nominal throttle that ~holds corner in a turn
    "k_throttle": 0.004,        # KCAS error → throttle
    "idle_throttle": 0.05,
    "mil_throttle": 0.95,       # SIM space [0,1]
    # Altitude = ENERGY for the turn. A hard turn trades altitude; that is correct BFM.
    # Spend it freely from the ~7000 m spawn down to a low emergency floor — do NOT
    # interrupt the turn at mid-altitude (the W0 self-vs-self failure: an override at
    # 4500 m killed the bank for most of the episode → no sustained turn → flew apart).
    "k_alt": 0.0,               # altitude-hold OFF: full pull for the break turn at any altitude
    "alt_hold_m": 5500.0,       # (unused while k_alt=0)
    "alt_floor_m": 1500.0,      # EMERGENCY floor only (well above 305 m loss; recovers — 0% crash @1130 m)
    "alt_ceiling_m": 9500.0,    # above: gentle nose-down
    "climb_pull": 0.75,
    # Per-episode randomization → a DISTRIBUTION of competent turners (anti-cheese)
    "jitter": {"turn_gain": 0.08, "lag_deg": 3.0, "alt_floor_m": 400.0},
    "react_lag_steps": 0,       # >0: hold action for N steps (reaction delay)
    "seed": 0,
    # ★ STRENGTH DIAL (260615): one scalar ∈[0,1] = the annealing lever (AOS-style
    # opponent-strength curriculum). None ⇒ use the explicit knobs above. When set it
    # DERIVES turn_gain/g_pull_cap/pull_clip/lag_deg/near_range_m/corner_kcas from one
    # number: 0 = weakest/widest/laggiest (learner out-turns it = beatable), 1 = tightest/
    # most aggressive, ~0.5 ≈ these W0-tuned defaults. The curriculum raises it on a
    # win-gate (foundation phase: start ~0.2, anneal up as the learner converts).
    "strength": None,
}


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _apply_strength(cfg: dict) -> dict:
    """Derive the beatability knobs from cfg['strength'] ∈[0,1] (the annealing lever).
    Anchored so strength≈0.5 reproduces the W0-tuned explicit defaults; strength itself
    takes precedence over any explicit beatability knob when set."""
    s = cfg.get("strength")
    if s is None:
        return cfg
    s = min(1.0, max(0.0, float(s)))
    cfg["turn_gain"]    = _lerp(0.40, 0.92, s)   # turn rate / max bank
    cfg["base_pull"]    = _lerp(0.10, 0.30, s)   # ★ baseline pull — MUST stay below g_pull_cap, else
                                                 #   pull DECREASES with turn_demand (inverted) AND a
                                                 #   high constant pull climbs→stalls→crashes at low str
    cfg["g_pull_cap"]   = _lerp(0.28, 0.55, s)   # sustained-turn tightness (always > base_pull)
    cfg["pull_clip"]    = _lerp(0.40, 0.65, s)   # absolute pull ceiling
    cfg["lag_deg"]      = _lerp(16.0,  2.0, s)   # aim-behind = beatability (big lag = easy to out-turn)
    cfg["near_range_m"] = _lerp(1000.0, 600.0, s)
    cfg["corner_kcas"]  = _lerp(380.0, 320.0, s) # weaker = faster = wider turn radius = beatable
    return cfg


def _merge(spec: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (spec or {}).items() if k != "jitter"})
    jit = dict(DEFAULTS["jitter"])
    jit.update((spec or {}).get("jitter") or {})
    cfg["jitter"] = jit
    cfg = _apply_strength(cfg)
    return cfg


class MergeTeacherProvider(ActionProvider):
    """Aggressive-but-beatable, self-crash-proof scripted merge opponent."""

    def __init__(self, spec: dict | None = None):
        self.spec = _merge(spec)
        seed = int(self.spec.get("seed", 0))
        # ★ 260809 REPRODUCIBILITY FIX: this was np.random.default_rng(abs(hash((seed, "merge_teacher"))) % 2**32).
        # Python randomizes hash() of str/tuple-containing-str per PROCESS unless PYTHONHASHSEED is set, so the
        # opponent's per-episode jitter drew from a DIFFERENT stream on every run -- identical bundle + identical
        # env seed did not reproduce. Measured on band_g1 vs jittered_teacher, same bundle, same seed 4321:
        # +73.6%/3 kills on one run vs +27.4%/1 kill on another (46 pt swing). Every jteacher-cell number this
        # project ever recorded carries that uncontrolled variance. zlib.crc32 over stable bytes is deterministic
        # across processes and interpreters; the per-episode variety (the point of the jitter) is unchanged.
        self._rng = np.random.default_rng(zlib.crc32(b"merge_teacher:%d" % seed))
        self._ep = dict(self.spec)        # per-episode (jittered) params
        self._cached: np.ndarray | None = None
        self._hold = 0

    # ── per-episode randomization ─────────────────────────────────────────
    def reset(self, context: ActionContext | None = None) -> None:
        jit = self.spec["jitter"]
        ep = dict(self.spec)
        ep["turn_gain"] = float(np.clip(
            self.spec["turn_gain"] + self._rng.uniform(-1, 1) * jit.get("turn_gain", 0.0),
            0.30, 0.95))
        ep["lag_deg"] = float(max(0.0,
            self.spec["lag_deg"] + self._rng.uniform(-1, 1) * jit.get("lag_deg", 0.0)))
        ep["alt_floor_m"] = float(
            self.spec["alt_floor_m"] + self._rng.uniform(-1, 1) * jit.get("alt_floor_m", 0.0))
        self._ep = ep
        self._cached = None
        self._hold = 0

    # ── control law ───────────────────────────────────────────────────────
    def compute_action(self, context: ActionContext) -> ActionResult:
        ep = self._ep
        lag = int(ep.get("react_lag_steps", 0))
        if self._cached is not None and lag > 0 and self._hold < lag:
            self._hold += 1
            return ActionResult(action=self._cached, source="merge_teacher_hold",
                                info={"opponent_id": "scripted:merge_teacher"})
        self._hold = 0

        obs = np.asarray(context.observation, dtype=np.float32).ravel()
        # PITCH sign (FighterSim.py:105): -1=aft(pull/nose-UP), +1=fwd(push/nose-DOWN).
        # A PULL is therefore a NEGATIVE command. pull_cmd(p>=0) → emit -pitch_sign*p.
        def pull_cmd(p):
            return -float(ep["pitch_sign"]) * float(p)

        if obs.size < 13:  # degenerate (no obs) → wings-level gentle-climb cruise
            act = np.array([0.0, pull_cmd(0.10), 0.0, ep["mil_throttle"]], np.float32)
            self._cached = act
            return ActionResult(action=act, source="merge_teacher",
                                info={"opponent_id": "scripted:merge_teacher"})

        # normalize() maps [min,max]→[-1,1]; asymmetric fields invert as (obs+1)*half+min.
        roll = float(obs[0]) * 180.0                 # [-180,180] symmetric
        speed = (float(obs[3]) + 1.0) * 300.0        # KCAS [0,600]
        alt = (float(obs[4]) + 1.0) * 7500.0         # ALT [0,15000]
        dn, de, dd = float(obs[6]) * 15000.0, float(obs[7]) * 15000.0, float(obs[8]) * 8000.0
        # ★ FIX (260615): use obs[11]=LOS_az (clean SIGNED body-frame azimuth), NOT
        # obs[9]=ATA. _get_antenna_train_angle(proj=False) returns arccos(...) ∈ [0,180]
        # — STRICTLY non-negative, so sign(ata) was ALWAYS +1 → the teacher banked ONE
        # fixed direction every merge and flew apart to ~10km (the "can't re-engage"
        # failure was this bug, not energy). LOS_az is the proper signed lateral bearing.
        brg = float(obs[11]) * 180.0         # signed bearing to learner (>0 ⇒ right)
        el = float(obs[12]) * 90.0           # learner elevation (>0 ⇒ above)
        dist = float((dn * dn + de * de + dd * dd) ** 0.5)

        turn_gain = ep["turn_gain"]
        phi_max = turn_gain * float(ep["max_bank_deg"])

        # Lateral: bank toward the learner. Lead (pure pursuit) far → turn in hard;
        # lag (aim behind) inside near_range → deliberately beatable.
        if dist >= float(ep["near_range_m"]):
            aim = brg
        else:
            # LAG (beatable): aim lag_deg BEHIND the learner on the SAME side, floored at
            # boresight — never cross over. The old `brg - lag*sign(brg)` banked AWAY for a
            # near-boresight learner (|brg|<lag) = close-range roll chatter (audit 260615).
            aim = float(np.sign(brg)) * max(0.0, abs(brg) - float(ep["lag_deg"]))
        phi_des = float(np.clip(ep["k_bank"] * aim, -phi_max, phi_max))
        roll_cmd = float(ep["roll_sign"]) * float(np.clip(
            ep["k_roll"] * (phi_des - roll), -1.0, 1.0))

        # Vertical: PULL to sustain the turn (a banked level turn needs load ~1/cos φ,
        # so pull rises with bank/turn_demand) + ALTITUDE-HOLD (pull more below the hold
        # band so a hard turn doesn't bleed the jet into the floor — the v2 descending-
        # spiral) + LOS-el track (el>0 target above ⇒ more pull). Pull is a magnitude;
        # pull_cmd emits it as the (negative=aft) elevator command.
        turn_demand = min(1.0, abs(aim) / 90.0)
        pull = ep["base_pull"] + (ep["g_pull_cap"] - ep["base_pull"]) * turn_demand
        pull += ep["k_alt"] * (float(ep["alt_hold_m"]) - alt)   # below hold band → more pull (climb)
        pull += ep["k_pitch_el"] * el
        pull = float(np.clip(pull, -0.20, ep["pull_clip"]))
        pitch_cmd = pull_cmd(pull)

        # Throttle: HOLD CORNER SPEED so the turn radius is tight enough to re-engage.
        # Proportional — idle when fast (bleed the merge speed down to corner), mil when
        # slow (don't depart). This is THE fix for "can't re-merge" (W0 v3): a fast jet
        # turns in a ~5 km circle; at corner (~340 KCAS) the radius is ~1 km.
        spd_err = speed - float(ep["corner_kcas"])
        throttle = float(np.clip(ep["cruise_throttle"] - ep["k_throttle"] * spd_err,
                                 ep["idle_throttle"], ep["mil_throttle"]))

        # SOFT altitude-floor override: shallow the bank to climb out (still contesting)
        # rather than fully leveling (which abandons the fight). Full climb only here.
        if alt < float(ep["alt_floor_m"]):
            roll_cmd *= 0.35                                                  # shallow bank, climb out
            pitch_cmd = pull_cmd(ep["climb_pull"])                            # climb = PULL up (negative)
            throttle = ep["mil_throttle"]
        elif alt > float(ep["alt_ceiling_m"]):
            pitch_cmd = max(pitch_cmd, 0.10)                                  # gentle nose-DOWN (fwd=+)

        act = np.array([roll_cmd, pitch_cmd, 0.0, float(throttle)], dtype=np.float32)
        self._cached = act
        return ActionResult(action=act, source="merge_teacher",
                            info={"opponent_id": "scripted:merge_teacher"})

    def close(self) -> None:
        return None


__all__ = ["MergeTeacherProvider", "DEFAULTS"]
