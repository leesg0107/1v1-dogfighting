# -*- coding: utf-8 -*-
"""Competition-rule-aligned reward (team01).

Structure follows the vendor default (validated lineage: damage differential
+ weak pursuit shaping + graded altitude penalty + sparse terminal), with the
terminal block corrected to the COMPETITION's judging semantics, which the
env default mishandles:

  1. Opponent flies below 1000 ft  → competition WIN  (env default: draw -30)
  2. Own crash                     → competition LOSS (env default: draw -30)
  3. Timeout (200 s)               → damage advantage decides the winner
                                     (env default: no terminal reward at all)

Magnitude policy (docs/training-strategy-research-ko.md §4): terminal ≈
100-200x the per-step shaping bound; no close-range damage bonus (PHANG-MAN
counter-vulnerability side effect); shaping terms keep vendor-tuned defaults.

Required contract:
  - MY_REWARD_CONFIG dict
  - compute_reward(...) -> (total: float, components: dict)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    # Per-step shaping (vendor-tuned defaults kept)
    "survival_bonus": 0.0,
    "step_penalty": -0.01,
    "pursuit_scale": 0.3,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "damage_scale": 20.0,
    # Graded altitude penalty: gentle gradient before the hard 300 m floor
    "low_altitude_penalty": 0.1,        # ALT < soft threshold
    "low_altitude_soft_m": 600.0,
    "very_low_altitude_penalty": 0.3,   # ALT < hard-warning threshold
    "low_altitude_hard_m": 450.0,
    # Terminal (competition semantics)
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "timeout_win_reward": 80.0,
    "timeout_loss_reward": -80.0,
    "disengage_as_loss": False,        # ★ True = drifting out to max_disengage = LOSS (close the chip-then-run harbor)
    "draw_reward": -10.0,
    # ★ 260703 ACQ-RATCHET (env-side, single_agent_env._compute_step_reward): farm-safe aim-closing gradient
    #   that fills the gun_snap dead band (ATA ~50→4°) so the ata_offset nose-off ACQUISITION completes. Monotone
    #   best-so-far ratchet on g=exp(-ATA/k); range/advantage/closure are HARD GATES. ALL default OFF = vendor-identical.
    "acq_scale": 0.0,                  # enable at ~12.0 in the experiment yaml (bounded ~12/ep, telescopes)
    "acq_k_deg": 18.0,                 # aim kernel exp(-ATA/k); alive across 50→4° (g 0.06→0.80)
    "acq_range_gate_m": 914.0,         # HARD gate = WEZ max range (bank only where a shot is possible; NOT a factor)
    "acq_adv_min": 0.20,               # rel_posture(proj=True) advantage gate = behind/favourable only (merge~0 = closed)
    "acq_close_min_mps": 0.0,          # SIGN gate: acq pays only when range is NOT opening (anti-fly-out). 0 = "not fleeing" (dt-robust). A >0 MAGNITUDE floor blocked the acquisition turn (gated 2-3/2000); the monotone ratchet already kills co-orbit
    "acq_decay_per_s": 0.0,            # ★ 260703 DECAY RATCHET: best_g decays at this rate (/s) so a LOSE→REGAIN cycle RE-PAYS (re-acquisition gradient). enable ~0.015-0.02 in the yaml. 0 = legacy monotone ratchet (nose-on lazy-init DEAD). Farm cap = acq_scale*this ≈ 0.24/s << damage 0.5-5/s
    "acq_ep_cap": 0.0,                 # ★ 260703 red-team: per-EPISODE hard cap on cumulative acq payout (farm backstop; true sup with decay ≈48, set ~42). 0 = no cap
    "guard_fail_penalty": -50.0,
    "timeout_health_margin": 0.02,
    # b1 (research 260614): score a no-damage timeout (the agent never entered
    # the WEZ) as a LOSS, not a draw — a STRUCTURAL (un-amortizable) close of the
    # safe-draw harbor that the standing engage penalty alone gets "eaten". Off by
    # default (vendor/competition fidelity); merge-training turns it on.
    "no_engage_timeout_as_loss": False,
    "no_engage_health_floor": 0.98,   # both healths above this at a tie = no engagement
    # Relative positional-advantage shaping (McGrew/LAG, 260614). None/0 = off.
    # posture = (1-|ATA|/180)·(1-|AA|/180); reward = scale·(mine − enemy's)·range_factor.
    "position_scale": 0.0,            # set ~0.2 to enable the BFM position gradient
    "position_range_m": 914.0,        # full posture weight at/inside this range (WEZ outer)
    "position_range_scale_m": 3000.0, # linear decay of range_factor beyond position_range_m
    # Steep posture form (reward-dilemma-audit 260615): exp(-|ATA|/k_ata)*exp(-|AA|/k_aa)
    # concentrates the gradient into the gun cone (the linear form was flat there).
    # Both >0 enables it (in position AND the env PBRS phi); 0 = legacy linear.
    "posture_k_ata": 0.0,             # set ~30 (deg) to enable steep aim gradient
    "posture_k_aa": 0.0,              # set ~50 (deg) to enable steep tail gradient
    # PBRS posture range-gate (farm-kill 260615, read by env._potential): 0 = ungated
    # (posture pays at ANY range → farmable at 3-5km, the n2merge position↑/WEZ↓ trap).
    # Set ~900 so the phi posture term only counts as the agent closes into the firing band.
    "pbrs_posture_range_m": 0.0,
    "pbrs_posture_range_scale_m": 1500.0,
    # WEZ-ENTRY latch (260615, read by env._compute_step_reward): a one-time bonus the first
    # step the agent reaches an ADVANTAGE firing solution (in-WEZ AND rel_posture≥adv_min —
    # the rear control zone, not a mutual head-on clip). The merge-conversion objective
    # ("reach the attack zone FROM the merge"); latched so it can't be dither-farmed. 0 = off.
    "wez_entry_bonus": 0.0,           # set ~40 for the merge/foundation phase
    "wez_entry_adv_min": 0.20,        # min (my_posture - enemy_posture) to count as advantage
    # ① ENGAGE axis (260614 disengagement fix): hold the WEZ firing band,
    # PENALIZE extension. The neutral failure was the agent extending to ~10 km
    # after the merge (TRUE WEZ 0%) — range was never a standalone priority,
    # only a factor coupled to aiming. This makes the close-fight envelope an
    # explicit "track" (racing-centerline analog). 0 = off.
    "engage_scale": 0.0,              # set ~0.1 to prioritize staying in firing range
    "engage_near_m": 250.0,           # ramp up below this (anti-collision/overshoot)
    "engage_far_m": 900.0,            # firing band outer (WEZ phase0 ~914 m)
    "engage_extend_scale_m": 1500.0,  # beyond far: linear → negative (anti-extension)
    # ── OUT-BOX (260718): CONTINUOUS anti-out-boxing = engage the WHOLE 200s (competition manual
    #   slide15: running far away to stall = penalty/loss). PURE per-step penalty, triple-gated
    #   (FAR and nose-AWAY and OPENING) so it integrates to cost ∝ time-spent-out-boxing. Farm-free
    #   (no positive term), barrel-in-safe (zero inside outbox_range_m — no proximity gradient),
    #   re-engagement-safe (the nose/closure gate exempts the turn-back + any far tail-chase; fly-apart
    #   is kinematic per flyapart-adjudication-260717). Saturates so it never swamps the ±80 terminal.
    #   Co-guards (must stay on): own_damage_scale 100 (a brawl still nets negative) + no margin gate
    #   (negative income, nothing to farm). Range >= lead_ratchet_range_hi 2500 so it never taxes the
    #   conversion lead-turn. 0 = off (vendor/legacy byte-identical).
    "outbox_penalty_weight": 0.0,     # set ~0.03; saturated -0.03/step ≈ -0.3/s (120s out-box ≈ -36/ep). Keep <=0.05.
    "outbox_range_m": 3000.0,         # zero penalty inside; ramp starts here (>= lead_ratchet_range_hi 2500)
    "outbox_ramp_m": 3000.0,          # linear 3km->6km then saturate flat to the 8km disengage terminal
    "outbox_ata_deg": 90.0,           # nose-away gate: penalize only |ATA|>90 (nose in the rear hemisphere)
    "outbox_closing_eps_mps": 0.0,    # opening gate: penalize only when closure < eps (range not shrinking)
    # Opponent flies into the ground ("target altitude below min").
    # None = competition semantics (full win_reward). Bootstrap phases override
    # this to 0.0: every vendor scripted target (BT ~61 s, loiter ~36 s)
    # self-crashes from our init geometries, and paying +100 for it makes
    # "fly safe and wait" the optimal policy (v1/v2 postmortem) — the agent
    # never discovers the gun. Self-play/eval restore None for rule fidelity.
    "opponent_ground_reward": None,
    # ── C-RECIPE (260620): published merge recipes — TempFuser energy (cubed alt + AoA) + PHANG-MAN Aggressive-Shooter ──
    # Cubed low-altitude penalty (TempFuser arXiv:2308.03257): ~0 high up (doesn't mask the objective) + extremely
    # STEEP near the deck = the exact form the linear/moderate guards lacked. penalty = -w*((onset-alt)/(onset-floor))^3. 0=off.
    "alt_cubed_weight": 0.0,           # set ~5
    "alt_cubed_onset_m": 2000.0,
    "alt_cubed_floor_m": 300.0,
    # AoA penalty (TempFuser): over-pull -> high AoA -> bleed -> stall = the self-crash CAUSE (untried). AoA=atan2(w,u). 0=off.
    "aoa_penalty_weight": 0.0,         # set ~0.03; penalty = -w*max(0,|AoA_deg|-thresh)
    "aoa_threshold_deg": 30.0,
    # Aggressive-Shooter gun-snap (PHANG-MAN arXiv:2105.00990): CONTINUOUS reward INCREASING as the agent closes
    # inside the firing band WHILE aimed -> pulls INTO the shot, not extend-to-reposition. reward = w*aim*close. 0=off.
    "gun_snap_weight": 0.0,            # set ~0.3
    "gun_snap_cone_deg": 8.0,          # aim_factor = max(0,1-ATA/cone)
    "gun_snap_max_m": 914.0,           # close_factor ramp TOP (WEZ max)
    "gun_snap_min_m": 152.4,           # ★ 260703 WEZ MIN: close_f=0 below it (the 0-damage dead zone) — kills the point-blank barrel-in pull
    "gun_snap_k_deg": 0.0,             # ★ >0 = STEEP 4th-power exp(-(ata/k)^4) (HHMARL); drives LOS<2° precision
    # Graded enemy-HP-depletion terminal (PHANG-MAN: no HP-depletion terminal -> gave away near-victory). At ep end,
    # reward proportional to HP dealt (1-target_health) -> finishing > extending. 0=off.
    "hp_depletion_weight": 0.0,        # set ~40
    # ── C2 LEAD-PURSUIT (260620): the MERGE OBJECTIVE (proportional-navigation / gun-lead). The 10th attempt = the
    # FIRST correct task framing. ATA-pursuit rewards "point at the CURRENT position" = pure pursuit = the OVERSHOOT
    # at high closure (the agent did exactly what we rewarded). LEAD rewards aligning the NOSE with the LEAD point
    # (P_tgt + V_tgt*t_lead), t_lead = DYNAMIC time-to-intercept (NOT a constant — the #1 trap). Straight target ->
    # reduces to pure pursuit (θ_lead≈ATA, rear PRESERVED); crossing target -> nose pulled AHEAD = the lead turn.
    # ★ TRAP 2: set pursuit_scale=0 (REPLACE ATA-pursuit, not add — else the old term fights lead -> overshoot stays).
    # ★ TRAP 3: first run = lead-swap ALONE (clean 1-change). If ata_behind drops but front-quarter win stays 0 ->
    #   THEN add a lag/AA-conversion term (lead fixes terminal turn-tracking; the head-on->six conversion may need lag).
    "lead_scale": 0.0,                # set ~0.5; reward = scale*exp(-theta_lead/k)
    "lead_k_deg": 20.0,
    "lead_closing_min_mps": 50.0,     # floor on closing speed (avoid huge t_lead when not closing)
    "lead_t_min_s": 0.5,
    "lead_t_max_s": 5.0,
    # ── C2-LAG CONVERSION (260620): the ANTI-OVERSHOOT objective. lag pursuit (nose BEHIND, bleed) prevents the
    # barrel-in/fly-past = the overshoot (MORE DIRECT than lead — cut-across can still overshoot at high closure).
    # ① reward getting BEHIND (low AA, range-gated); ② PENALIZE high closure when close (anti-barrel-in -> bleed/lag);
    # ③ ATA aim ONLY at terminal (AA low + in firing range, else ATA-aim re-triggers overshoot). pursuit_scale=0.
    # ★★ GATE: lag keeps ATA HIGH during conversion -> "ata_behind DOWN" is the WRONG gate; judge "AA reaches low AND HOLDS".
    "lag_scale": 0.0,                 # set ~1.0
    "lag_k_aa_deg": 50.0,             # ① exp(-AA/k_aa) = occupy the six
    "lag_aa_range_m": 2500.0,         # ① range-gate (within engagement range)
    "lag_overshoot_w": 0.02,          # ② -w*relu(closure)*relu(1-range/R_ctrl)
    "lag_ctrl_range_m": 1500.0,       # ② control range (closer = stronger anti-overshoot)
    "lag_overshoot_aa_gate_deg": 0.0, # ② AA-gate: 0=always; ~45 = only fire when AA>gate (not yet behind) so the agent CLOSES once behind
    "lag_close_w": 0.0,               # ④ close-when-behind (AA<=gate): set ~0.5
    "lag_close_mode": "range",        # ④ 'lead' = LEAD pursuit (cut inside the turn, the fix) | 'range' = positional pull (failed)
    "lag_close_k_deg": 30.0,          # ④ lead mode: w*exp(-theta_lead/k)
    "lag_close_range_m": 1500.0,      # ④ range mode
    "lag_term_lead": False,           # ③ terminal: True = aim at the LEAD point (cut inside a turn) | False = current-position ATA
    "lag_term_w": 0.5,                # ③ terminal aim +w*exp(-ATA/k) when AA<aa_term AND range<term_range
    "lag_term_aa_deg": 40.0,
    "lag_term_k_ata_deg": 8.0,
    "lag_term_range_m": 914.4,
}


def _closure_mps(ownship_state, target_state):
    """Closing speed (m/s, positive = range shrinking). Uses the GIVEN body velocities (state[6:9])."""
    o = np.asarray(ownship_state, dtype=float)
    t = np.asarray(target_state, dtype=float)
    v_o = _R_nb(o[StateIndex.ROLL:StateIndex.YAW + 1]).T @ o[StateIndex.U:StateIndex.W + 1]
    v_t = _R_nb(t[StateIndex.ROLL:StateIndex.YAW + 1]).T @ t[StateIndex.U:StateIndex.W + 1]
    r = t[StateIndex.N:StateIndex.D + 1] - o[StateIndex.N:StateIndex.D + 1]
    rng = float(np.linalg.norm(r))
    if rng < 1e-6:
        return 0.0
    return float((v_o - v_t) @ (r / rng))


def _R_nb(rpy_deg):
    """NED->body DCM (GeoMathUtil convention T_nb = tx@ty@tz), matching observation.py kinematics."""
    d2r = math.pi / 180.0
    phi, theta, psi = d2r * float(rpy_deg[0]), d2r * float(rpy_deg[1]), d2r * float(rpy_deg[2])
    tx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(phi), math.sin(phi)], [0.0, -math.sin(phi), math.cos(phi)]])
    ty = np.array([[math.cos(theta), 0.0, -math.sin(theta)], [0.0, 1.0, 0.0], [math.sin(theta), 0.0, math.cos(theta)]])
    tz = np.array([[math.cos(psi), math.sin(psi), 0.0], [-math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    return tx @ ty @ tz


def lead_pursuit_angle(ownship_state, target_state, cfg):
    """θ (deg) between the ownship NOSE and the bearing to the LEAD point (P_tgt + V_tgt*t_lead), with t_lead =
    DYNAMIC time-to-intercept = clamp(range / max(closing, v_min), t_min, t_max). Returns (theta_deg, t_lead).
    Straight target -> theta≈ATA (pure pursuit reduction = rear preserved); crossing target -> nose pulled AHEAD."""
    o = np.asarray(ownship_state, dtype=float)
    t = np.asarray(target_state, dtype=float)
    Ro = _R_nb(o[StateIndex.ROLL:StateIndex.YAW + 1])
    Rt = _R_nb(t[StateIndex.ROLL:StateIndex.YAW + 1])
    p_o = o[StateIndex.N:StateIndex.D + 1]
    p_t = t[StateIndex.N:StateIndex.D + 1]
    v_o = Ro.T @ o[StateIndex.U:StateIndex.W + 1]          # ownship velocity in NED
    v_t = Rt.T @ t[StateIndex.U:StateIndex.W + 1]          # target velocity in NED
    r = p_t - p_o
    rng = float(np.linalg.norm(r))
    if rng < 1e-6:
        return 0.0, 0.0
    u_los = r / rng
    closing = float((v_o - v_t) @ u_los)                  # >0 = closing
    v_min = float(cfg.get("lead_closing_min_mps", 50.0))
    t_min = float(cfg.get("lead_t_min_s", 0.5))
    t_max = float(cfg.get("lead_t_max_s", 5.0))
    t_lead = min(t_max, max(t_min, rng / max(closing, v_min)))   # ★ DYNAMIC (trap 1)
    lead_pt = p_t + v_t * t_lead
    lb = lead_pt - p_o
    lbn = float(np.linalg.norm(lb))
    nose = Ro.T @ np.array([1.0, 0.0, 0.0])               # ownship forward (gun line) in NED
    if lbn < 1e-6:
        return 0.0, t_lead
    cosang = max(-1.0, min(1.0, float(nose @ (lb / lbn))))
    return math.degrees(math.acos(cosang)), t_lead


def _terminal_reward(
    cfg: dict,
    ownship_health: float,
    target_health: float,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> float:
    def health_judge(win_value: float, loss_value: float) -> float:
        margin = float(cfg.get("timeout_health_margin", 0.02))
        if ownship_health > target_health + margin:
            # ★ anti-TURTLE (red-team 260627): the flat win_value is a STEP — a 2% chip earns the SAME +80 as a
            #   near-kill, so the optimal policy becomes "chip 2% once then high-altitude turtle". timeout_win_
            #   proportional scales the win by the ACTUAL damage-margin so MORE chip = MORE reward (= the real
            #   competition objective). Default off (no change to other configs).
            if bool(cfg.get("timeout_win_proportional", False)):
                full = float(cfg.get("timeout_win_full_margin", 0.5))   # margin earning the full win_value
                frac = max(0.0, min(1.0, (ownship_health - target_health) / max(1e-6, full)))
                return win_value * frac
            return win_value
        if ownship_health < target_health - margin:
            return loss_value
        # Tie. b1: a no-damage tie = a passive no-show (never entered the WEZ).
        # Scoring it as a LOSS closes the safe-draw harbor structurally.
        if bool(cfg.get("no_engage_timeout_as_loss", False)):
            floor = float(cfg.get("no_engage_health_floor", 0.98))
            if ownship_health >= floor and target_health >= floor:
                return loss_value
        return float(cfg.get("draw_reward", -10.0))

    if terminated:
        if end_condition == "two circle headon guard fail":
            return float(cfg.get("guard_fail_penalty", -50.0))
        if target_health <= 0.0 < ownship_health:
            return float(cfg.get("win_reward", 100.0))
        if ownship_health <= 0.0 < target_health:
            return float(cfg.get("loss_reward", -100.0))
        # Competition rule: below 1000 ft = crash = that side loses.
        if end_condition == "target altitude below min":
            override = cfg.get("opponent_ground_reward")
            if override is not None:
                return float(override)
            return float(cfg.get("win_reward", 100.0))
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            return float(cfg.get("loss_reward", -100.0))
        # ★ DISENGAGE harbor (260702 DEFECT-1 FIX): drifting out to max_disengage_range was falling through to
        #   health_judge below and COLLECTING 80*margin (~58) at ~30s = the free "chip-then-flee" settlement
        #   (verified: 97.9% of eps end disengage-draw; under gamma 0.995 the ~24s early payout is ~5500x the
        #   present-value of the same margin at the 200s timeout, so fleeing STRICTLY dominates staying). Now a
        #   disengage pays disengage_reward (default draw_reward = 0): leaving banks NO margin, so the ONLY route
        #   to margin is to STAY to the 200s timeout. disengage_as_loss=true still routes to loss_reward (a
        #   stronger push if wanted; -100 is the terminal-swamp precedent from margin_lstm — avoid).
        if end_condition == "disengaged":
            if bool(cfg.get("disengage_as_loss", False)):
                return float(cfg.get("loss_reward", -100.0))
            return float(cfg.get("disengage_reward", cfg.get("draw_reward", 0.0)))
        # Mutual kill / fuel fail / other rare endings → damage advantage.
        return health_judge(
            float(cfg.get("timeout_win_reward", 80.0)),
            float(cfg.get("timeout_loss_reward", -80.0)),
        )
    if truncated:
        # Competition rule: at 200 s the side that dealt more damage wins.
        return health_judge(
            float(cfg.get("timeout_win_reward", 80.0)),
            float(cfg.get("timeout_loss_reward", -80.0)),
        )
    return 0.0


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    cfg = reward_config
    components: dict[str, float] = {}

    # 0. Survival bonus (curriculum compatibility; 0 by default)
    components["survival"] = float(cfg.get("survival_bonus", 0.0))

    # 1. Step penalty (time pressure)
    components["step"] = float(cfg.get("step_penalty", -0.01))

    # 2. Pursuit shaping: smooth ATA x range gradient (vendor formula)
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = abs(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    half_angle = float(cfg.get("pursuit_half_angle_deg", 30.0))
    pursuit_range = float(cfg.get("pursuit_range_m", 3000.0))
    ata_factor = max(0.0, 1.0 - ata / half_angle)
    range_factor = max(0.0, 1.0 - distance / pursuit_range)
    components["pursuit"] = float(cfg.get("pursuit_scale", 0.3)) * ata_factor * range_factor

    # 3. Damage differential (the core objective signal)
    # ★ 260713 R3-B ASYMMETRIC MARGIN: own_damage_scale defaults to damage_scale (= the
    #   original SYMMETRIC 50*(target-own), byte-identical when the key is unset), but can be
    #   raised so getting HIT costs strictly more than dealing damage. WHY: the symmetric term
    #   let a mutual brawl (deal 90/take 49 = net +20.5) OUT-EARN a clean no-hit occupation
    #   (deal 20/take 0 = +10) -> PPO drifted the v4d2r seed from own-dmg 0.1% to 11.3% (merge
    #   margin +10 -> +1.2). This is the literal encoding of the user's vision: occupy the rear
    #   WITHOUT getting hit. At own_damage_scale=100 the same brawl nets 45-49 = -4 (negative)
    #   while clean stays +10. Aim/kill capability is untouched (the target term is unchanged).
    _dmg_scale = float(cfg.get("damage_scale", 20.0))
    _own_dmg_scale = float(cfg.get("own_damage_scale", _dmg_scale))
    components["damage"] = _dmg_scale * float(target_damage) - _own_dmg_scale * float(ownship_damage)

    # 4. Safety: two-tier low-altitude gradient ahead of the hard floor
    altitude = float(ownship_state[StateIndex.ALT])
    safety = 0.0
    if altitude < float(cfg.get("low_altitude_hard_m", 450.0)):
        safety = -float(cfg.get("very_low_altitude_penalty", 0.3))
    elif altitude < float(cfg.get("low_altitude_soft_m", 600.0)):
        safety = -float(cfg.get("low_altitude_penalty", 0.1))
    # C-RECIPE: CUBED low-altitude penalty (TempFuser) — ~0 high up (objective unmasked), extremely STEEP near deck.
    cw = float(cfg.get("alt_cubed_weight", 0.0))
    if cw != 0.0:
        onset = float(cfg.get("alt_cubed_onset_m", 2000.0))
        floor = float(cfg.get("alt_cubed_floor_m", 300.0))
        if altitude < onset:
            frac = min(1.0, max(0.0, (onset - altitude) / max(1.0, onset - floor)))
            safety -= cw * frac ** 3
    components["safety"] = safety

    # 4b. C-RECIPE AoA penalty (TempFuser): over-pull -> high AoA -> energy bleed -> stall = the self-crash CAUSE.
    aoa_w = float(cfg.get("aoa_penalty_weight", 0.0))
    if aoa_w != 0.0:
        u = float(ownship_state[StateIndex.U]); w = float(ownship_state[StateIndex.W])
        aoa_deg = abs(math.degrees(math.atan2(w, u)))
        components["aoa"] = -aoa_w * max(0.0, aoa_deg - float(cfg.get("aoa_threshold_deg", 30.0)))
    else:
        components["aoa"] = 0.0

    # 4b'. Control-smoothness (260702 viz-driven): the deterministic-MEAN policy PORPOISES (eval alt 650→9800m
    #      swings, action_sat 0.8 bang-bang) = uncontrolled vertical oscillation, and never sustains a track
    #      (in-WEZ 3-5%, final_ata rode to 167° = departed). Penalize sustained high body PITCH-RATE |Q| beyond
    #      a deadband. FARM-FREE (pure penalty — no positive to game). DEFAULT weight 0 = OFF (σ-hygiene is the
    #      clean step-1 lever; enable ~0.005 only if the porpoise survives entropy 0.001 + log_std_clip 0.6).
    pr_w = float(cfg.get("pitch_rate_penalty_weight", 0.0))
    if pr_w != 0.0:
        q_degps = abs(float(ownship_state[StateIndex.Q]))
        q_dead = float(cfg.get("pitch_rate_deadband_degps", 30.0))
        components["smooth"] = -pr_w * max(0.0, q_degps - q_dead)
    else:
        components["smooth"] = 0.0

    # 4c. C-RECIPE Aggressive-Shooter gun-snap (PHANG-MAN): reward INCREASES as the agent closes inside the firing
    # band WHILE aimed -> pulls INTO the shot, deters the extend-to-reposition (the diagnosed overshoot/drift).
    gs_w = float(cfg.get("gun_snap_weight", 0.0))
    if gs_w != 0.0:
        cone = float(cfg.get("gun_snap_cone_deg", 8.0))
        gmax = float(cfg.get("gun_snap_max_m", 914.0))
        gmin = float(cfg.get("gun_snap_min_m", 152.4))   # ★ 260703 WEZ MIN (dead zone below): NO reward under it
        # ★ STEEP gun-cone (260628, viz-driven): the path viz showed the agent gets CLOSE (45% in-range) but
        # NEVER tightens to a gun solution (LOS<2deg) -> 0 damage. The LINEAR aim (1-ata/cone) and the wide LEAD
        # exp(-theta/20) are FLAT near 0 = no gradient to tighten 10deg->2deg. gun_snap_k_deg>0 uses the HHMARL
        # 4th-power exp(-(ata/k)^4): ~0.04 @10deg, ~0.99 @2deg = a STRONG gradient to drive the final precision.
        gk = float(cfg.get("gun_snap_k_deg", 0.0))
        if gk > 0.0:
            aim_f = math.exp(-((ata / gk) ** 4))
        else:
            aim_f = max(0.0, 1.0 - ata / cone) if cone > 0.0 else 0.0
        # ★ 260703 WEZ-ALIGNED close ramp (was max(0,1−r/gmax) which PEAKED at r=0 = the point-blank DEAD ZONE
        #   where competition damage=0, so gun_snap paid MAX where a shot deals 0 → the agent rationally barreled
        #   in to 11m and could not aim [ω=v/r blows up]). Now close_f MIRRORS the competition damage ramp exactly:
        #   (gmax−r)/(gmax−gmin) inside [gmin,gmax], ZERO below the WEZ min (152.4m) and beyond gmax. The point-blank
        #   pull is gone; the product aim_f×close_f self-finds the skill-dependent optimal range (aim_f→0 where the
        #   angular rate is un-aimable), so no hand-coded anti-barrel shape is needed. single_agent_env.py:996-999.
        close_f = (gmax - distance) / max(1e-6, gmax - gmin) if gmin <= distance <= gmax else 0.0
        components["gun_snap"] = gs_w * aim_f * close_f
    else:
        components["gun_snap"] = 0.0

    # 4d. C2 LEAD-PURSUIT (the MERGE objective; pursuit_scale=0 REPLACES the pure-pursuit ATA term). Reward aligning
    # the NOSE with the dynamic LEAD point -> the lead turn at high closure instead of "point at them" -> overshoot.
    lead_scale = float(cfg.get("lead_scale", 0.0))
    if lead_scale != 0.0:
        theta_lead, _t_lead = lead_pursuit_angle(ownship_state, target_state, cfg)
        components["lead"] = lead_scale * math.exp(-theta_lead / float(cfg.get("lead_k_deg", 20.0)))
    else:
        components["lead"] = 0.0

    # 4e. C2-LAG conversion (anti-overshoot): ① get BEHIND (low AA) ② PENALIZE high closure when close (no fly-past
    # = forces bleed/lag) ③ ATA aim ONLY at the terminal (AA low + in range; else aiming re-triggers the overshoot).
    lag_scale = float(cfg.get("lag_scale", 0.0))
    if lag_scale != 0.0:
        aa = abs(float(geo_info._get_aspect_angle(ownship_state, target_state, True)))  # 260630 vision-redesign: proj=True (collapse-free 2D aspect, six=0/head-on=180); was False (3D sign-hack collapses head-on->~0 = paid full rear salary at a head-on merge). occupy-six must require ACTUALLY being behind.
        # ① occupy the six (low AA), range-gated to engagement range
        lag = math.exp(-aa / float(cfg.get("lag_k_aa_deg", 50.0))) if distance < float(cfg.get("lag_aa_range_m", 2500.0)) else 0.0
        # ② anti-overshoot: penalize high closing speed when close (prevents barrel-in/fly-past). ★ AA-GATED
        # (260620 fix): fire ONLY during the conversion (AA > gate). Once BEHIND (AA <= gate), turn OFF so the
        # agent CLOSES to the kill — else "lag-and-hold" is the reward optimum (converts but never finishes).
        aa_gate = float(cfg.get("lag_overshoot_aa_gate_deg", 0.0))   # 0 = always (legacy); ~45 = only when not yet behind
        if aa_gate <= 0.0 or aa > aa_gate:
            r_ctrl = float(cfg.get("lag_ctrl_range_m", 1500.0))
            closeness = max(0.0, 1.0 - distance / r_ctrl)
            closure = max(0.0, _closure_mps(ownship_state, target_state))
            lag -= float(cfg.get("lag_overshoot_w", 0.02)) * closure * closeness
        else:
            # ④ once BEHIND: LEAD pursuit (cut INSIDE the turn) to close+tighten (260621). Range-closeness FAILED
            # (a TURNING target keeps the lag radius constant -> "get closer" never fires; you must cut INSIDE the
            # turn = lead). Reward exp(-theta_lead/k) once behind -> the agent leads -> closes -> ③ shot -> kill.
            # lag_close_mode 'lead' = the fix; else 'range' = the (failed) positional pull. 0 weight = off.
            w_close = float(cfg.get("lag_close_w", 0.0))
            if w_close != 0.0:
                if str(cfg.get("lag_close_mode", "range")) == "lead":
                    _tl2, _ = lead_pursuit_angle(ownship_state, target_state, cfg)
                    lag += w_close * math.exp(-_tl2 / float(cfg.get("lag_close_k_deg", 30.0)))
                else:
                    lag += w_close * max(0.0, 1.0 - distance / float(cfg.get("lag_close_range_m", 1500.0)))
        # ③ terminal aim ONLY when BEHIND (AA low) AND in firing range. ★ lag_term_lead (260621): aim at the LEAD
        # point (proportional-navigation), NOT current-position ATA — a TURNING target needs LEAD at the terminal
        # (cut INSIDE the turn to tighten the kill); pure-pursuit ATA keeps lagging the turn -> never closes the kill.
        if aa < float(cfg.get("lag_term_aa_deg", 40.0)) and distance < float(cfg.get("lag_term_range_m", 914.4)):
            if bool(cfg.get("lag_term_lead", False)):
                _theta_lead, _ = lead_pursuit_angle(ownship_state, target_state, cfg)
                lag += float(cfg.get("lag_term_w", 0.5)) * math.exp(-_theta_lead / float(cfg.get("lag_term_k_ata_deg", 8.0)))
            else:
                lag += float(cfg.get("lag_term_w", 0.5)) * math.exp(-ata / float(cfg.get("lag_term_k_ata_deg", 8.0)))
        components["lag"] = lag_scale * lag
    else:
        components["lag"] = 0.0

    # 6. Positional advantage (RELATIVE posture: aim ATA + tail AA, zero-sum).
    # Fixes the ATA-only flaw (research 260614: McGrew/LAG/PHANG-MAN). The old
    # geometry reward used ATA (aim) only, so head-on — where BOTH aim at each
    # other — paid both sides, producing the mutual no-engage draw (even
    # BT-vs-BT = 200s, 0 damage). Posture multiplies aim by TAIL position:
    #   posture(o,t) = (1-|ATA|/180)·(1-|AA|/180)
    # Head-on: aim low but AA~180 (not behind) → posture~0. Behind+aimed → ~1.
    # Made RELATIVE (mine − enemy's) so head-on is symmetric → net 0 (no reward
    # for the mutual merge); positive ONLY by getting to the enemy's six while
    # they don't get yours. This is the dogfight analog of racing's centerline
    # PROGRESS: a dense gradient toward the control position, not just "face it".
    # range_factor pulls the engagement into the WEZ band. Off by default.
    pos_scale = float(cfg.get("position_scale", 0.0))
    if pos_scale != 0.0:
        # Steep McGrew/LAG posture form (reward-dilemma-audit 260615). The linear
        # (1-|ATA|/180) factor has a CONSTANT, tiny slope (1/180 per deg) → it goes
        # flat at coarse aim, giving almost no pull from ATA 45°→4° where the
        # |ATA|≤4° damage cliff and the whole win live. PPO had to STUMBLE the last
        # degrees (sporadic-then-forgotten kills). exp(-|ATA|/k) concentrates the
        # gradient into the gun cone (LAG reciprocal/arctanh analog). k_*=0 ⇒ linear.
        k_ata = float(cfg.get("posture_k_ata", 0.0))
        k_aa = float(cfg.get("posture_k_aa", 0.0))
        steep = k_ata > 0.0 and k_aa > 0.0
        # ATA range-gate (260616): reward AIM (ATA) only when CLOSE (< position_ata_range_m). On
        # the FAR merge approach, reward ONLY behind-ness (AA) + closure — NOT "aim now" — to
        # remove the PURE-PURSUIT incentive that converges to the OVERSHOOT (ATAbehind stuck~33°,
        # regressing). The diagnosed reward-landscape fix (best-ever 16° proved capability exists).
        # 0 = off (ATA rewarded at all ranges, the legacy pure-pursuit-prone form).
        ata_gate = float(cfg.get("position_ata_range_m", 0.0))
        ata_on = (ata_gate <= 0.0) or (distance <= ata_gate)
        def _posture(o, t):
            ata_o = abs(float(geo_info._get_antenna_train_angle(o, t, False)))
            aa_o = abs(float(geo_info._get_aspect_angle(o, t, False)))
            if steep:
                af = math.exp(-ata_o / k_ata) if ata_on else 1.0
                return af * math.exp(-aa_o / k_aa)
            af = (max(0.0, 1.0 - ata_o / 180.0) if ata_on else 1.0)
            return af * max(0.0, 1.0 - aa_o / 180.0)
        wez_max = float(cfg.get("position_range_m", 914.0))
        rng_scale = float(cfg.get("position_range_scale_m", 3000.0))
        rf = 1.0 if distance <= wez_max else max(0.0, 1.0 - (distance - wez_max) / rng_scale)
        components["position"] = pos_scale * (
            _posture(ownship_state, target_state) - _posture(target_state, ownship_state)
        ) * rf
    else:
        components["position"] = 0.0

    # 7. ENGAGE (anti-disengage): penalize EXTENSION beyond the WEZ firing band.
    # Diagnosis 260614 (success_existence_probe): in neutral the agent reaches the
    # enemy's six 100% of episodes but at dist@peak ~10 km — it EXTENDS after the
    # merge, so aim and range never coincide (TRUE WEZ 0%).
    # ★ 260614 fix: the in-band positive is REMOVED. An absolute-distance positive
    # here is farmable — a mutual ~700 m orbit pays +full to BOTH sides (+240/ep),
    # netting positive for a non-damaging orbit = the comfort-orbit attractor the
    # term was meant to kill. Now in-band (and closer) pays NOTHING; only extension
    # beyond d_far is penalized (linear → −1). Damage + terminal are the only
    # positives for being in the band → leaving is strictly worse than staying,
    # with no standing reward for loitering.
    eng_scale = float(cfg.get("engage_scale", 0.0))
    if eng_scale != 0.0:
        d_far = float(cfg.get("engage_far_m", 900.0))
        ext = float(cfg.get("engage_extend_scale_m", 1500.0))
        if distance <= d_far:
            prox = 0.0
        else:
            prox = max(-1.0, -(distance - d_far) / ext)
        components["engage"] = eng_scale * prox
    else:
        components["engage"] = 0.0

    # 4b. OUT-BOX (continuous anti-out-boxing; competition rule + engage-the-WHOLE-200s).
    #   PURE per-step penalty for sustained disengagement: FAR and nose-AWAY and OPENING. Integrates to
    #   cost ∝ time-out-boxing. Stateless (pure geometry) so it lives here. distance/ata computed above.
    ob_w = float(cfg.get("outbox_penalty_weight", 0.0))
    if (
        ob_w != 0.0
        and distance > float(cfg.get("outbox_range_m", 3000.0))
        and ata > float(cfg.get("outbox_ata_deg", 90.0))
        and _closure_mps(ownship_state, target_state) < float(cfg.get("outbox_closing_eps_mps", 0.0))
    ):
        over = min(1.0, (distance - float(cfg.get("outbox_range_m", 3000.0)))
                   / max(1.0, float(cfg.get("outbox_ramp_m", 3000.0))))
        components["outbox"] = -ob_w * over
    else:
        components["outbox"] = 0.0

    # 5. Terminal: competition judging semantics
    components["terminal"] = _terminal_reward(
        cfg,
        float(ownship_state[StateIndex.HEALTH]),
        float(target_state[StateIndex.HEALTH]),
        terminated,
        truncated,
        end_condition,
    )

    # 5b. C-RECIPE graded HP-depletion terminal (PHANG-MAN): at ep end, reward HP DEALT (1-target_health) so a
    # NEAR-kill is worth something -> the agent FINISHES instead of extending-to-reposition (the diagnosed failure).
    hp_w = float(cfg.get("hp_depletion_weight", 0.0))
    if hp_w != 0.0 and (terminated or truncated):
        components["hp_terminal"] = hp_w * max(0.0, 1.0 - float(target_state[StateIndex.HEALTH]))
    else:
        components["hp_terminal"] = 0.0

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
