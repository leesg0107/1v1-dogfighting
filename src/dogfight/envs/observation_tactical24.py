"""tactical24 / tactical24_fv observation (Team01 additions).

Extracted from the organizer-provided `dogfight/envs/observation.py`, which is not included here.
`normalize` and `_build_tactical16` come from that original module."""
from __future__ import annotations

import numpy as np

from dogfight.envs.observation import _build_tactical16, normalize
from dogfight.sim.state_schema import StateIndex

# ★ 260713 HISTORY-AUGMENTED obs (tactical24_h): tactical24 (24 ch) + a strided tail of PAST signed
#   LOS az/el. WHY: a memoryless policy imitating the stateful (latched) merge teacher MODE-AVERAGES
#   the teacher's left-OR-right reversal into an indecisive ~0 (drift → 42% draws, measured). A short
#   az/el history breaks that aliasing (which side was the target on before the ±180 wrap → which way
#   to reverse). Deploy-safe: a ring-buffer of recent frames, NO hidden state. The tail is appended by
#   the ENV in get_observation() (NOT in build_observation) so pool opponents stay on 24 ch.
#   TAC24_H_LAGS taps spaced TAC24_H_STRIDE RL-steps apart → tail length = 2 * TAC24_H_LAGS.
TAC24_H_LAGS = 2      # number of past taps (az, el each)
TAC24_H_STRIDE = 4    # RL-steps between taps (~0.1s/step → taps at ~0.4s and ~0.8s ago)


def tac24_h_history_maxlen() -> int:
    """deque(maxlen=...) size a caller must use for the tactical24_h history ring-buffer."""
    return TAC24_H_LAGS * TAC24_H_STRIDE


def augment_tac24_h_tail(base, geo_info, ownship_state, target_state, history, push: bool = True) -> "np.ndarray":
    """Append the tactical24_h strided history tail to a 24-ch BASE obs and push the current LOS.

    SINGLE SOURCE OF TRUTH for the env (single_agent_env.get_observation) AND the deploy path
    (unreal/policies.py) — the deployed tail MUST match training bit-for-bit. `history` is a
    caller-owned collections.deque(maxlen=tac24_h_history_maxlen()) cleared once per episode; call
    this ONCE per policy decision (== once per RL step, matching the env cadence). Tail = signed LOS
    az/el at t-STRIDE, t-2*STRIDE, ... (zero-padded early in the episode). ★ push=False reads the tail
    WITHOUT advancing the ring-buffer (for a re-build on a sub-step where the decision is cached — the
    opponent obs is rebuilt step_ratio times per RL step but must push exactly once).
    """
    az, el = geo_info._get_los_angle(ownship_state, target_state)
    cur = (normalize(float(az), -180.0, 180.0), normalize(float(el), -90.0, 90.0))
    hist = list(history)   # oldest→newest
    tail = []
    for lag in range(TAC24_H_STRIDE, TAC24_H_STRIDE * TAC24_H_LAGS + 1, TAC24_H_STRIDE):
        idx = len(hist) - lag
        tail.extend(hist[idx] if idx >= 0 else (0.0, 0.0))
    if push:
        history.append(cur)
    return np.concatenate([np.asarray(base, dtype=np.float32), np.asarray(tail, dtype=np.float32)])


# ★ 260714 FULL-VECTOR frame-stack (tactical24_fv): tactical24 (24) + strided taps of the WHOLE 24-ch
#   base at t-STRIDE, t-2*STRIDE = 72 dim. WHY: the finish skill was knife-edge FRAGILE under the
#   tactical24_h LOS-only tail (measured: a 0.001% weight drift flipped 17->13 wins). A full-vector
#   stack lets the MLP finite-diff EVERY channel (target turn-rate, closure-rate, g-onset) itself, so
#   the finish decision becomes a robust wide-basin function of observable state instead of a fragile
#   weight memorization. Deploy-safe by construction: SAME ring-buffer pattern as tactical24_h, shared
#   by the env AND unreal/policies (train==deploy bit-for-bit). Uses ONLY deploy-available channels
#   (the wire lacks body-rates/Nz/AoA — the stack derives 2nd-order from what deploy actually sends).
TAC24_FV_LAGS = 2
TAC24_FV_STRIDE = 4

# ★ 260718 DEEP full-vector frame-stack (tactical24_fv4): the representation-lever prototype. fv reaches
#   only 0.8s (t-4,t-8) but a merge REVERSAL commit spans ~2s (the |los_az|>60 turn-around, avg 151deg,
#   that caps conversion — 260713r/q: the ceiling is opponent-INDEPENDENT = representation-bound). fv4
#   adds taps to 1.6s (t-4,t-8,t-12,t-16 = 24 + 4*24 = 120-dim) to disambiguate reversal-direction states
#   the 2-tap stack still mode-averages. SAME deploy-safe ring-buffer contract; fv is unchanged.
TAC24_FV4_LAGS = 4
TAC24_FV4_STRIDE = 4

# per-mode (lags, stride) for the full-vector family; fv is byte-identical to before.
_FV_PARAMS = {
    "tactical24_fv": (TAC24_FV_LAGS, TAC24_FV_STRIDE),
    "tactical24_fv4": (TAC24_FV4_LAGS, TAC24_FV4_STRIDE),
}


def tac24_fv_history_maxlen() -> int:
    return TAC24_FV_LAGS * TAC24_FV_STRIDE


def _augment_fv_tail_params(base, history, lags, stride, push: bool = True) -> "np.ndarray":
    """Full-vector history tail with explicit (lags, stride). fv/fv4 both route here."""
    base = np.asarray(base, dtype=np.float32)
    hist = list(history)   # oldest→newest; each item = a 24-vec base snapshot
    zero = np.zeros(base.shape[0], dtype=np.float32)
    tail = []
    for lag in range(stride, stride * lags + 1, stride):
        idx = len(hist) - lag
        tail.append(hist[idx] if idx >= 0 else zero)
    if push:
        history.append(base.copy())
    return np.concatenate([base] + tail)


def augment_tac24_fv_tail(base, geo_info, ownship_state, target_state, history, push: bool = True) -> "np.ndarray":
    """Full-vector analogue of augment_tac24_h_tail: push the WHOLE 24-ch base and append strided taps.

    SINGLE SOURCE OF TRUTH for env + deploy (unreal/policies) — same contract as augment_tac24_h_tail
    (call ONCE per decision; `history` is a per-episode deque(maxlen=tac24_fv_history_maxlen())). Each
    history item is the full 24-vec base; the tail is base(t-STRIDE), base(t-2*STRIDE), zero-padded early.
    ★ push=False reads the tail without advancing the ring-buffer (opponent sub-step re-build guard).
    """
    return _augment_fv_tail_params(base, history, TAC24_FV_LAGS, TAC24_FV_STRIDE, push=push)


_HISTORY_MODES = ("tactical24_h", "tactical24_fv", "tactical24_fv4")


def is_history_mode(mode: str) -> bool:
    """True if the mode appends a get_observation()-time history tail (env + deploy both must augment)."""
    return mode in _HISTORY_MODES


def history_maxlen(mode: str) -> int:
    """deque(maxlen=...) for a history mode's ring-buffer (content differs by mode)."""
    if mode in _FV_PARAMS:
        lags, stride = _FV_PARAMS[mode]
        return lags * stride
    return tac24_h_history_maxlen()


def augment_history_tail(mode, base, geo_info, ownship_state, target_state, history, push: bool = True) -> "np.ndarray":
    """Mode-aware dispatch: LOS-pair tail (tactical24_h) vs full-vector tail (tactical24_fv / _fv4)."""
    if mode in _FV_PARAMS:
        lags, stride = _FV_PARAMS[mode]
        return _augment_fv_tail_params(base, history, lags, stride, push=push)
    return augment_tac24_h_tail(base, geo_info, ownship_state, target_state, history, push=push)


def _build_tactical17(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """tactical16 + a high-resolution boresight ATA feature (index 16).

    Coarse ATA at obs[9]=normalize(ata,-180,180): 1°=0.0056, so the ±1°
    competition kill cone is a sub-resolution sliver — this capped gun-kill
    learning (±4°:1.0, ±2°:0.74, ±1°:~0). obs[16]=normalize(ata,-10,10) gives
    1°=0.1 (18x finer near boresight) so the policy can perceive ±1°.
    260613 observation-resolution finding.
    """
    base = _build_tactical16(ownship_state, target_state, geo_info, wez_config)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    fine = np.empty(17, dtype=np.float32)
    fine[:16] = base
    fine[16] = normalize(float(ata), -10.0, 10.0)
    return fine


def _R_nb(rpy_deg):
    """NED->body DCM, IDENTICAL convention to GeoMathUtil._get_los_angle (T_nb = tx@ty@tz)."""
    d2r = np.pi / 180.0
    phi, theta, psi = d2r * float(rpy_deg[0]), d2r * float(rpy_deg[1]), d2r * float(rpy_deg[2])
    tx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), np.sin(phi)], [0.0, -np.sin(phi), np.cos(phi)]])
    ty = np.array([[np.cos(theta), 0.0, -np.sin(theta)], [0.0, 1.0, 0.0], [np.sin(theta), 0.0, np.cos(theta)]])
    tz = np.array([[np.cos(psi), np.sin(psi), 0.0], [-np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    return tx @ ty @ tz


def _kinematic_features(ownship_state, target_state):
    """The 'line' features from the GIVEN body-frame velocities (state[6:9], P1-verified real).

    Returns (rel_vel_body[3] m/s in own body frame, los_rate_az deg/s, los_rate_el deg/s, range_rate m/s).
    LOS-rate sign matches GeoMathUtil az=arctan2(body[1],body[0]) / el=-arcsin(body[2]).
    """
    own = np.asarray(ownship_state, dtype=float)
    tgt = np.asarray(target_state, dtype=float)
    R_own = _R_nb(own[3:6])
    R_tgt = _R_nb(tgt[3:6])
    v_own_ned = R_own.T @ own[6:9]    # own body-vel -> NED
    v_tgt_ned = R_tgt.T @ tgt[6:9]    # tgt body-vel -> NED
    v_rel_ned = v_tgt_ned - v_own_ned
    v_rel_body = R_own @ v_rel_ned     # relative velocity in OWN body frame
    r_ned = tgt[0:3] - own[0:3]
    rng = float(np.linalg.norm(r_ned))
    if rng < 1e-6:
        return v_rel_body, 0.0, 0.0, 0.0
    u_los = r_ned / rng
    range_rate = float(v_rel_ned @ u_los)              # d||r||/dt; negative = closing
    v_perp = v_rel_ned - range_rate * u_los            # rel-vel perpendicular to LOS
    vp_body = R_own @ v_perp
    r2d = 180.0 / np.pi
    los_rate_az = float(vp_body[1] / rng) * r2d        # deg/s
    los_rate_el = float(-vp_body[2] / rng) * r2d       # deg/s
    return v_rel_body, los_rate_az, los_rate_el, range_rate


def _build_tactical23(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """tactical17 + 6 'line' kinematic features (rel_vel_body 3, LOS-rate az/el 2, closure 1) from the
    GIVEN body-frame velocity — lets a SINGLE frame anticipate the bandit's motion (lead pursuit)
    instead of reacting to its current point. The deferred-then-indicated obs upgrade for conversion."""
    obs = np.empty(23, dtype=np.float32)
    obs[:17] = _build_tactical17(ownship_state, target_state, geo_info, wez_config)
    rvb, los_az, los_el, range_rate = _kinematic_features(ownship_state, target_state)
    obs[17] = normalize(float(rvb[0]), -600.0, 600.0)       # rel_vel_body x (fwd)
    obs[18] = normalize(float(rvb[1]), -600.0, 600.0)       # rel_vel_body y (right)
    obs[19] = normalize(float(rvb[2]), -600.0, 600.0)       # rel_vel_body z (down)
    obs[20] = normalize(float(los_az), -40.0, 40.0)         # LOS-rate az deg/s (merge band; the lead signal)
    obs[21] = normalize(float(los_el), -40.0, 40.0)         # LOS-rate el deg/s
    obs[22] = normalize(float(-range_rate), -600.0, 600.0)  # closure m/s (positive = closing)
    return obs


def _build_tactical24(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """tactical23 + the TARGET bank angle (roll) = a single-frame TURN indicator (omega ~ g*tan(phi)/V,
    sign = turn direction). tac16/17/23 DROPPED the target's roll (they use only AA/LOS = heading-derived,
    NOT bank); relative14 had it (target_state[ROLL]). The merge FINISH needs to ANTICIPATE the target's
    CONTINUING turn (curvature = 2nd-order) to lead the cone — tac23 gives only 1st-order (velocity, LOS-rate).
    Bank supplies the missing turn rate in ONE frame, no memory. The cheapest test of the anticipation
    hypothesis (the lag-conversion finish gap: gets behind ~28deg, can't tighten on a turning target)."""
    obs = np.empty(24, dtype=np.float32)
    obs[:23] = _build_tactical23(ownship_state, target_state, geo_info, wez_config)
    obs[23] = normalize(float(target_state[StateIndex.ROLL]), -180.0, 180.0)  # target bank = single-frame turn indicator
    return obs


