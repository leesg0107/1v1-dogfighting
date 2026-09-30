# 1v1 Dogfighting — TOPGUN 2026 AI Pilot Challenge (Team01)

Core logic of a reinforcement-learning agent for a one-on-one, guns-only F-16 dogfight.

## Competition and simulator

- **Task.** 1v1 within-visual-range gun fight, 200-second rounds. The side that deals more damage wins, or a kill ends the round.
- **Weapon engagement zone.** Damage accrues only while the target sits inside a narrow cone around the nose at 500–3,000 ft range: 1° half-angle for the first 100 s, widening to 2° and then 3° later in the round.
- **Altitude.** Main rounds start at 2,000–3,000 ft (610–914 m); the crash floor is 1,000 ft.
- **Simulator.** The organizer provided the training environment: JSBSim F-16 flight dynamics compiled into Windows DLLs, Python wrappers, and a Gym-style environment for Ray RLlib. Official matches run on the organizer's battle server. The submitted agent is a UDP client that receives aircraft state packets and returns four raw stick commands (roll, pitch, yaw, throttle). We developed on Linux and ran the Windows simulator under Wine through a UDP bridge.

**Why this repository does not run on its own.** The simulator, its wrappers and the base environment belong to the organizer and were given to participants without a redistribution license, so they are left out. This repository contains only the code we wrote. It shows how the agent works but needs the organizer's environment to run.

## Result

Four wins and two losses in the Swiss-format preliminaries. Eliminated.

The agent reliably got behind its opponents but could not hold the gun on target long enough to finish them. After the event we also found a deployment bug: in the submitted build, throttle was converted to simulator range before the braking layers ran, so every brake command was sent as −0.70 instead of 0.15.

## Final model

A hybrid of one neural network and three rule-based layers.

| Layer | Type | Responsibility | Code |
|---|---|---|---|
| Policy | PPO (RLlib, MLP 256×256, 181k parameters) | Roll, pitch, rudder | `src/dogfight/ai/residual_module.py` |
| GCAS | Rule | Ground-collision avoidance | `src/dogfight/ai/gcas.py` |
| Throttle governor | Rule (closure-rate latch) | Caps throttle on fast approaches to prevent overshoot | `src/dogfight/ai/throttle_governor.py` |
| Terminal takeover | Rule (lead-pursuit law) | Takes all four channels when the target is within 20° of the nose and 1,500 m | `src/dogfight/ai/terminal_takeover.py` |

- **Observation.** `tactical24_fv`, 72 dimensions: a 24-channel tactical vector (own attitude, speed and altitude; relative position; aspect and line-of-sight angles and rates; closure; target state) plus two past frames. See `src/dogfight/envs/observation_tactical24.py`.
- **Action.** Raw stick, four dimensions in [−1, 1], at 10 Hz (60 Hz simulation, action repeat 6).

## How we got there

1. **Pure RL (failed).** We spent about a month on reward shaping and curricula, and the agent never learned to win the head-on merge. There were two reasons:
   - **Sparse reward.** Damage only accrues inside a 1° cone, so outside it the learning signal is zero. Kills take 50–1,200 decisions, far longer than the horizon PPO could credit. Each shaping term we added to fill the gap was either exploited or, once gated, stopped teaching anything.
   - **Compute and time.** Everything ran on one workstation (one RTX 4090, 16 CPU cores). The Windows simulator ran under Wine, which limited throughput. Our largest single run was about 35 million environment steps, and all runs together came to about 0.8 billion. Heron Systems won the 2020 DARPA AlphaDogfight Trials with pure end-to-end RL, but reportedly trained on billions of dogfights (about 4 billion training examples) over roughly five weeks. That is two orders of magnitude more experience per agent than we could afford in two months.
2. **Scripted teacher and imitation.** We wrote a rule-based teacher (reverse, intercept, track; `student/selfplay/autopilot_teacher.py`) and distilled it with behavior cloning and four DAgger rounds (`student/tools/bc_pretrain.py`).
3. **History in the observation.** A memoryless policy averaged the teacher's left-or-right decisions into indecision. Adding past frames (`tactical24_fv`) cut the draw rate from 42% to 29%.
4. **Residual PPO.** Plain PPO fine-tuning destroyed the imitated skills. We froze the base network and trained only a bounded residual head (`residual_module.py`, `anchor_ppo_learner.py`), then adapted it to competition altitude (`experiments/ours_ppo_r11_lowalt.yaml`).
5. **Anchor distillation → `p1d1`.** We ran DAgger with a teacher that combined the RL policy with a scripted brake. The network could not learn the brake, so throttle control moved into a separate rule (the governor).
6. **Terminal takeover (the main gain).** We split the kill chain into measurable stages (`student/eval/funnel_probe.py`) and found a single broken link: holding the aim. Handing the final phase to a scripted law **raised the kill rate from 0.31% to 14.8%** over 264 episodes, with no losses of our own aircraft.
7. **Competition-altitude retraining (no improvement).** We fixed an altitude drift and an opponent observation bug, then retrained at competition altitude. The result did not beat the existing model.

| Kill-chain stage (760 m spawn) | Network alone | Scripted teacher |
|---|---|---|
| Opportunities per minute | 1.30 | 1.40 |
| Damage taken | 3.1% | 12.4% |
| Time inside the 1° cone per episode | 0.13 s | 0.51 s |
| Damage per opportunity | 1.4 pt | 8.0 pt |

Every stage up to the shot was at parity or better. Only the final aim was missing.

## Lessons

**Reward the shot, not the approach.** Damage only counts inside a 1° cone, so a reward based on damage alone is almost always zero and the agent gets no learning signal. To fix that, we added dense rewards for good approach geometry: getting behind the target, aligning for a lead turn, and entering the engagement zone. These terms were weighted 3–12. The only term that rewarded keeping the nose on target had weight 1 and faded to zero beyond about 4° off the nose. The agent learned what it was paid for. It became good at reaching a firing position and poor at firing from it. The reward should stay close to the real score (damage dealt minus damage taken), and any aim shaping must be strong and wide enough to guide the agent from a rough aim to a precise one.

**Leave precise terminal aim to a control law.** Holding a 1° cone takes continuous fine correction, plus discrete, latched decisions such as when to brake and which way to reverse. A memoryless network acting at 10 Hz struggles with both. When it imitates a teacher's either-or decision, it averages the two options into something in between. Every network we trained held the cone for only 0.05–0.26 s per episode, against 0.51 s for a simple lead-pursuit law. Reward changes, extra network heads and more training did not close that gap. Moving the final phase to the control law did. RL works well for maneuvering into position; the last few degrees are better handled by classical guidance.

## Repository layout

```
src/dogfight/ai/
  terminal_takeover.py      terminal aim takeover (lead-pursuit law)
  throttle_governor.py      approach throttle governor
  gcas.py                   predictive ground-collision avoidance
  residual_module.py        frozen base network + bounded residual head
  anchor_ppo_learner.py     PPO anchored to the imitation policy
  asym_logstd.py            asymmetric log-std clipping
src/dogfight/envs/
  observation_tactical24.py tactical24 / tactical24_fv observation
student/
  my_reward.py              reward function
  selfplay/                 scripted teacher and scripted opponents
  tools/bc_pretrain.py      behavior cloning / DAgger pretraining
  eval/funnel_probe.py      kill-chain funnel diagnostics
experiments/
  ours_ppo_r11_lowalt.yaml  low-altitude (760 m) PPO config
```

## References

- C. R. DeMay, E. L. White, W. D. Dunham, J. A. Pino, "AlphaDogfight Trials: Bringing Autonomy to Air Combat," *Johns Hopkins APL Technical Digest* 36(2), pp. 154–163, 2022. Heron Systems' winning agent and its self-play training setup.
- A. P. Pope et al., "Hierarchical Reinforcement Learning for Air-to-Air Combat," ICUAS 2021, arXiv:2105.00990. Lockheed Martin's second-place hierarchical agent.
- J. H. Bae, H. Jung, S.-H. Kim, S. Kim, Y.-D. Kim, "Deep Reinforcement Learning-Based Air-to-Air Combat Maneuver Generation in a Realistic Environment," *IEEE Access* 11, pp. 26427–26440, 2023. Raw-stick control with a recurrent SAC policy and a reverse curriculum.
- J. Chai, W. Chen, Y. Zhu, Z.-X. Yao, D. Zhao, "A Hierarchical Deep Reinforcement Learning Framework for 6-DOF UCAV Air-to-Air Combat," *IEEE Trans. SMC: Systems* 53(9), pp. 5417–5429, 2023.
- C. Chen, T. Song, L. Mo, M. Lv, D. Lin, "Autonomous Dogfight Decision-Making for Air Combat Based on Reinforcement Learning with Automatic Opponent Sampling," *Aerospace* 12(3), 265, 2025. A single-GPU training budget for 1v1 dogfighting.
- A. Selmonaj et al., "Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent RL," arXiv:2510.11474, 2025.
- S. Li et al., "An Imitative Reinforcement Learning Framework for Pursuit-Lock-Launch Missions," arXiv:2406.11562. Imitation combined with RL for air combat.
- Q. Liu, Y. Jiang, X. Ma, *Light Aircraft Game (CloseAirCombat)*, JSBSim-based 1v1 air-combat RL benchmark, github.com/liuqh16/CloseAirCombat.
