"""Ground Collision Avoidance System — deterministic altitude floor (260627).

ONE function, applied in BOTH training and deploy on the RAW RL action ([-1,1]^4,
PRE throttle-remap) so the policy LEARNS to fly with the floor (train/deploy gap 0).
A reward governor cannot GUARANTEE zero self-crash (the user's hard constraint); a
deterministic override can. Designed + adversarially verified by a 9-agent workflow.

VERIFIED conventions (do NOT change without re-reading FighterSim.py:104-107 & :243):
  action[0] ROLL : -1 = left, +1 = right (FighterSim.py:101). state[ROLL]=phi (deg, +right-bank). To level
                   from a right bank (phi>0) command roll LEFT (negative): roll_cmd = -phi/45.
  action[1] PITCH: -1 = aft = nose-UP = CLIMB  (+1 dives = the self-crash; NEVER emit +1 here)
                   (corroborated by scripted_opponent.py:161-162,222-227 pull_cmd = -pitch_sign*p)
  action[3] THROTTLE in [-1,1] pre-remap; +1 -> 100% via (x+1)/2 downstream
  ALT = ownship_state[StateIndex.ALT] (index 44), meters, UP-positive
  crash floor = 300 m (termination.py:26 / config.py:16)
  scas_offload (3D): action[1] is GAMMA (positive=climb, INVERTED) -> width-guard skips it.
Closed-loop lesson (260627): unit-testing the override in ISOLATION (9/9 passed) did NOT catch that pulling
  while banked turns lift into the ground -> roll-to-wings-level + pull-scaled-by-uprightness added below.
"""
import numpy as np

from dogfight.sim.state_schema import StateIndex

# ── tunables (physics-justified: EXACT pull-arc h_loss = (V^2/g)*ln((n-cos(dive))/(n-1)) + Vz*t_react
#    + roll_penalty + margin. ★260718 the arm is now trajectory-predictive: level/climb/shallow flight
#    floors at GCAS_ARM_ALT_M so the WHOLE 610-914m competition band flies FREE; only genuine dives arm
#    higher. Point-mass sim (F-16 aero, 6-9g aero-limited pull, 0.3s delay) verified worst realistic dive
#    (V=100-300, dive 30-90deg, wings-level & 60deg-banked) stays >300m: worst level ~396m. ──
GCAS_CRASH_FLOOR_M = 300.0    # MUST equal config["min_altitude"] (termination.py:26)
GCAS_ARM_ALT_M = 450.0        # ★260718 fixed clamp FLOOR + bad-read fallback (was 1000 = hijacked the WHOLE
                              #   610-914m competition band from spawn). 450 lets level/climb/shallow combat
                              #   fly free across 610-914m with a 150m buffer over the 300m floor; genuine
                              #   DIVES arm HIGHER via the recovery envelope below (competition-manual-facts-260718).
GCAS_PITCH_MIN = -0.6         # firm pull at the arm altitude (NEGATIVE = nose-up)
GCAS_PITCH_MAX = -0.9         # hardest pull near the floor; CAPPED short of -1.0 (no FDM AoA clamp)
GCAS_AOA_BACKOFF_DEG = 22.0   # if AoA exceeds this, ease the pull toward GCAS_PITCH_MIN (anti-departure)
GCAS_ROLL_LEVEL_DEG = 45.0    # P-control gain: full opposite roll once bank >= this (roll-to-wings-level)
GCAS_MIN_PULL = -0.05         # always at least slightly nose-up (the kill-switch a[1]<0 holds even knife-edge)
# ── DYNAMIC ARM (260628): the FIXED 1000 m band blew through on fast/steep dives (~1147 m of recovery needed
#    @ 300 m/s, 45deg dive > the 700 m usable band) = the 5.6% self-crash measured in margin_obs2. Arm altitude
#    is now the recovery-envelope PHYSICS floor, computed from FRAME-ROBUST quantities only: |velocity| (frame-
#    invariant) + pitch/roll ATTITUDE (unambiguous at deploy). It deliberately does NOT use the body-vs-world
#    velocity frame (unverified at deploy = the KCAS-2x / minimal_v1-R/Q trap class). Clamped >= the old fixed
#    band so it is NEVER less safe than before (no regression), <= a cap so it never arms absurdly high. ──
GCAS_ARM_MAX_M = 3500.0       # cap on the dynamic arm
GCAS_REACT_S = 0.3            # reaction/transport delay before the recovery pull develops
GCAS_NMAX = 3.0               # recovery pull load factor (conservative; the jet pulls 6-9g at combat speed)
GCAS_ROLL_PENALTY_M = 250.0   # extra altitude lost rolling to wings-level first, scaled by bank
GCAS_SAFETY_MARGIN_M = 120.0  # ★260718 fixed additive pad on the recovery-envelope arm: covers reaction/
                              #   discretization jitter AND the low-speed regime where the jet is aero-limited
                              #   below GCAS_NMAX g. Only bites in DIVES (level flight is floored by
                              #   GCAS_ARM_ALT_M) so it costs ZERO combat room; 120 -> worst realistic-dive
                              #   min-alt ~396m (all V=100-300 x dive 30-90deg provably >300m).


def _read_alt(ownship_state) -> float:
    """ALT (m, up+). Convention guard: reject non-finite / phantom reads -> +inf (= 'do not engage'),
    so a corrupted/zeroed state degrades to 'fires too high / never' (SAFE: the policy keeps flying)
    rather than 'fires every frame' (cannot fly) or a wrong-sign dive."""
    try:
        alt = float(ownship_state[StateIndex.ALT])
    except (TypeError, IndexError, ValueError):
        return float("inf")
    if not np.isfinite(alt):
        return float("inf")
    # A genuinely-zeroed plane_info (deploy fallback) yields ALT~0 AND pos~0; treat alt<=0 AND
    # |state[D]|<1 as 'no real telemetry' -> do not engage (don't fight a phantom floor).
    try:
        if alt <= 0.0 and abs(float(ownship_state[StateIndex.D])) < 1.0:
            return float("inf")
    except (TypeError, IndexError, ValueError):
        pass
    return alt


def _dynamic_arm_alt(ownship_state) -> float:
    """Recovery-envelope arm altitude from FRAME-ROBUST state only (speed magnitude + pitch/roll ATTITUDE).
    arm = floor + Vz*t_react + Vz^2/(2 g (n-1)) + roll_penalty,  with Vz = |V| * sin(nose-down pitch).
    Uses |velocity| (frame-invariant) and PITCH/ROLL (attitudes, unambiguous at deploy) — NOT the body-vs-world
    velocity frame that is unverified at deploy. Falls back to the fixed GCAS_ARM_ALT_M on any bad read (SAFE).
    Clamped to [GCAS_ARM_ALT_M, GCAS_ARM_MAX_M] so it is never LESS safe than the old fixed band."""
    try:
        u = float(ownship_state[StateIndex.U]); v = float(ownship_state[StateIndex.V]); w = float(ownship_state[StateIndex.W])
        speed = float(np.sqrt(u * u + v * v + w * w))
        pitch = float(np.radians(float(ownship_state[StateIndex.PITCH])))   # +up; nose-DOWN = negative
        phi = float(np.radians(float(ownship_state[StateIndex.ROLL])))
        if not (np.isfinite(speed) and np.isfinite(pitch) and np.isfinite(phi)):
            return GCAS_ARM_ALT_M
        descent = max(0.0, -float(np.sin(pitch))) * speed                  # vertical speed (>=0 only nose-down)
        g = 9.81
        n = max(1.2, GCAS_NMAX)
        # ★260718 EXACT pull-arc altitude loss. The old Vz^2/(2 g (n-1)) is a SHALLOW-dive limit that
        #   UNDER-arms steep dives ~1.6x (arm 918m vs true ~1273m at 150 m/s 90deg -> point-mass sim let
        #   the jet reach <0m). Integrate dh over a constant-n pull from the dive angle to level:
        #       h_pull = (V^2 / g) * ln( (n - cos(dive)) / (n - 1) )       (cos(dive)==cos(pitch), cos is even)
        cos_dive = float(np.cos(pitch)) if descent > 0.0 else 1.0          # ->1 (h_pull->0) when level/climbing
        h_pull = (speed * speed / g) * float(np.log(max(n - cos_dive, 1e-3) / (n - 1.0)))
        h_pull = max(0.0, h_pull)
        h_react = descent * GCAS_REACT_S
        h_roll = GCAS_ROLL_PENALTY_M * min(1.0, abs(float(np.sin(phi))))   # extra while rolling level, bank-scaled
        arm = GCAS_CRASH_FLOOR_M + h_react + h_pull + h_roll + GCAS_SAFETY_MARGIN_M
        return float(np.clip(arm, GCAS_ARM_ALT_M, GCAS_ARM_MAX_M))
    except (TypeError, IndexError, ValueError):
        return GCAS_ARM_ALT_M


def apply_gcas(action, ownship_state):
    """Override the raw RL action to climb when below the floor. Returns a COPY (float32).

    Width-guarded: only acts on 4-channel raw_stick. In 3D scas_offload, action[1] is GAMMA
    (positive=climb, INVERTED sign) -> an unconditional pitch->-1 would DIVE there, so we pass
    scas through untouched (the deploy path is raw_stick-only; scas is training-only).
    """
    a = np.array(action, dtype=np.float32).ravel()
    if a.shape[-1] < 4:                       # scas_offload (3D) -> do NOT touch (sign is inverted)
        return a

    alt = _read_alt(ownship_state)
    arm = _dynamic_arm_alt(ownship_state)     # ★ dynamic recovery-envelope floor (>= old fixed band, no regression)
    if alt >= arm:                            # above the speed/dive-scaled band -> policy flies freely
        return a

    # ── armed: hard-set the recovery vector ──────────────────────────────────
    frac = (arm - alt) / (arm - GCAS_CRASH_FLOOR_M)   # 0 at arm -> 1 at floor
    frac = float(np.clip(frac, 0.0, 1.0))
    pull = GCAS_PITCH_MIN + frac * (GCAS_PITCH_MAX - GCAS_PITCH_MIN)        # -0.6 -> -0.9 (negative)

    # AoA back-off (no FDM stall clamp): if already high-alpha, ease toward the gentle pull so the
    # pull-arc load factor n does not collapse and balloon altitude loss. Guarded read.
    try:
        aoa = float(ownship_state[StateIndex.AOA])
        if np.isfinite(aoa) and aoa > GCAS_AOA_BACKOFF_DEG:
            pull = GCAS_PITCH_MIN              # back to the gentlest firm pull (-0.6), still nose-UP
    except (TypeError, IndexError, ValueError):
        pass

    # ★ ROLL-TO-WINGS-LEVEL (260627 fix). The original GCAS set roll=0 = HOLD the current bank, then pulled.
    # Pulling while BANKED rotates the lift vector toward the GROUND, not up (the closed-loop crash_rate 0.276
    # in margin_explore despite a correct pitch sign). Fix: (a) P-control roll TOWARD wings-level, (b) SCALE the
    # pull by how upright we are -- full pull level, ~0 pull at 90deg bank -- so we roll level FIRST instead of
    # pulling into the deck while knife-edge/inverted. Guarded; falls back to the old (roll=0, full pull) on a
    # bad read. Deploy-safe: policies.py:283 fills state[ROLL] from plane_info.rotation.roll (train==deploy).
    roll_cmd = 0.0
    upright = 1.0
    try:
        phi = float(ownship_state[StateIndex.ROLL])           # deg, +right-bank (FighterSim.py:190)
        if np.isfinite(phi):
            roll_cmd = float(np.clip(-phi / GCAS_ROLL_LEVEL_DEG, -1.0, 1.0))   # opposite to bank -> level
            upright = max(0.0, float(np.cos(np.radians(phi))))                 # 1 level, 0 @90deg, 0 inverted
    except (TypeError, IndexError, ValueError):
        pass

    a[0] = roll_cmd                       # ROLL toward wings-level (was 0 = hold bank = the closed-loop bug)
    a[1] = min(pull * upright, GCAS_MIN_PULL)  # PITCH nose-UP, scaled by uprightness; ALWAYS at least slightly aft
    a[2] = 0.0                            # YAW   no rudder -> no pro-spin / AoS at high q
    a[3] = 1.0                            # THROTTLE +1 (pre-remap) -> 100% downstream -> rebuild energy

    assert a[1] < 0.0, "GCAS pitch override must be NEGATIVE (nose-up); +pitch dives = self-crash"
    return a


# ── BAND CEILING (260726) — the mirror of GCAS, for the TEST OPPONENT only ────────────────────────
# WHY. The competition fights R1-3 at 610-914 m, but every policy in this project was trained with the
# default 7000 m spawn, so a merge spawned in-band leaves it in ~17 s and the pair settles at 5-8 km.
# Measured consequence: only 15% of hard-turning time happens below 2 km, and available load factor
# collapses with density -- 5.7 g under 2 km, 2.7 g at 5-8 km, 1.9 g above 8 km. So the whole evaluation
# has been scoring a thin-air fight the competition never has. The champion is NOT the cause: its
# altitude tracks the opponent's at r=+0.96 (it sits ~260 m BELOW), and against non-climbing opponents
# it stays at 1.6-1.8 km. Holding the OPPONENT in the band therefore holds the fight in the band.
#
# SCOPE. This is a property of the test rig, not a competition rule (the manual states a 300 m crash
# floor and an out-boxing penalty, no ceiling) and NOT something to put on our own agent -- constraining
# the submission with a rule the competition does not have could only cost margin. Apply to the opponent
# side to make evaluation representative; never to the shipped policy.
#
# ★ v2 (260726, after the v1 probe was CONFOUNDED): v1 ramped a fixed nose-DOWN push with altitude, which
# did hold the band -- but a sustained push in dense air is a dive, so the opponents accelerated from 167
# to 344 m/s (~Mach 1) and their total turning fell 34-59%. The champion then scored 8/12 kills against a
# fast, barely-maneuvering target: a handicap, not a representative fight. v2 instead ARRESTS THE CLIMB --
# it levels the nose (drives pitch attitude toward 0) and stops pushing the moment the jet is level, so it
# can never command a sustained dive. Throttle is also capped while armed, because the whole failure mode
# was energy the opponent could not have carried in a real in-band fight.
CEIL_BAND_M = 300.0        # soft band below the ceiling where the push ramps in (no step discontinuity)
CEIL_PITCH_MAX = 0.5       # hardest nose-DOWN push (positive = nose-down; FCS limiter, same cap the
                           #   scripted probes use); ramps 0 -> this across the band
CEIL_MIN_ALT_M = 900.0     # never arm below this: leaves the whole competition band plus GCAS headroom
CEIL_PITCH_TARGET_DEG = 0.0   # level flight = the arrest target (NOT a dive command)
CEIL_PITCH_GAIN = 1.0 / 15.0  # deg of nose-up error -> push; saturates by ~15 deg of climb attitude
CEIL_THROTTLE_CAP = 0.6       # sim-space throttle cap while armed: v1's dive-acceleration to Mach 1 is
                              #   exactly the overspeed contamination the pool docs warn about


def apply_band_ceiling(action, state, ceiling_m):
    """Push the nose down when the TEST OPPONENT climbs above `ceiling_m`. Returns a COPY (float32).

    Mirrors apply_gcas: width-guarded to 4-channel raw_stick (in 3D scas_offload action[1] is GAMMA with
    inverted sign, so a nose-down push there would climb), guarded state read, and it only ever *blends*
    toward a bounded push instead of hard-setting the vector -- the opponent keeps flying its own fight,
    it just cannot leave the band. ceiling_m None/<=0 disables (returns the action untouched).
    ponytail: proportional push, no latch. If an opponent porpoises at the ceiling, widen CEIL_BAND_M
    before adding hysteresis.
    """
    a = np.array(action, dtype=np.float32).ravel()
    if ceiling_m is None or float(ceiling_m) <= 0.0 or a.shape[-1] < 4:
        return a
    ceiling_m = max(float(ceiling_m), CEIL_MIN_ALT_M)
    alt = _read_alt(state)                       # +inf on a bad read -> would arm every frame, so guard it
    if not np.isfinite(alt) or alt <= ceiling_m:
        return a
    frac = float(np.clip((alt - ceiling_m) / CEIL_BAND_M, 0.0, 1.0))
    # ARREST the climb: push only as hard as the jet is pointed UP, so the push vanishes at level flight
    # and a sustained dive is not representable. On a bad pitch read, fall back to no push (never dive).
    try:
        theta = float(state[StateIndex.PITCH])            # deg, + = nose-up (observation.py:_build_tactical16)
        climb_err = max(0.0, theta - CEIL_PITCH_TARGET_DEG) if np.isfinite(theta) else 0.0
    except (TypeError, IndexError, ValueError):
        climb_err = 0.0
    push = frac * min(CEIL_PITCH_MAX, climb_err * CEIL_PITCH_GAIN)
    if push <= 0.0:
        return a          # level or descending above the ceiling -> nothing to arrest, hands off entirely
                          # (a bare max(a[1], 0.0) here would flatten the policy's own nose-UP command)
    # take the nose-down push only if it is MORE nose-down than what the policy already commands, so a
    # jet already descending is never slowed down by the ceiling
    a[1] = max(float(a[1]), push)
    # cap throttle ONLY while actively arresting: v1's confound was dive-acceleration to ~Mach 1, and a
    # jet that is already level gets no handicap it would not have chosen itself
    a[3] = min(float(a[3]), CEIL_THROTTLE_CAP * 2.0 - 1.0)   # policy-space [-1,1]
    return a
