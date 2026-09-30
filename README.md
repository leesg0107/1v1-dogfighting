# 1v1 Dogfighting — TOPGUN 2026 AI Pilot Challenge (Team01)

F-16 1:1 근접 기총 공중전(JSBSim, 200초, 기총 전용)을 위한 강화학습 에이전트의 핵심 로직입니다.
결과는 예선 스위스 라운드 4승 2패로 탈락했습니다.

> 이 레포에는 직접 작성한 핵심 코드만 들어 있습니다. 주최측이 제공한 시뮬레이터(DLL, 래퍼)와 학습 환경은 포함하지 않았으므로 단독으로는 실행되지 않습니다.

## 최종 모델

**`p1d1` 신경망 + 규칙 기반 레이어 3개로 된 하이브리드 구조**

| 레이어 | 방식 | 역할 | 코드 |
|---|---|---|---|
| 정책 | PPO (RLlib, MLP 256×256, 18만 파라미터) | roll / pitch / rudder | `src/dogfight/ai/residual_module.py` |
| GCAS | 규칙 | 지면 충돌 회피 | `src/dogfight/ai/gcas.py` |
| 스로틀 거버너 | 규칙 (closure-EMA 래치) | 고속 접근 시 스로틀 제한, 오버슛 방지 | `src/dogfight/ai/throttle_governor.py` |
| 터미널 인수 | 규칙 (리드 추적 법칙) | ATA < 20°, 거리 < 1,500 m 이면 4채널 전부 담당 | `src/dogfight/ai/terminal_takeover.py` |

- 관측: `tactical24_fv` 72차원 (전술 벡터 24채널 + stride-4 히스토리 탭 2개), `src/dogfight/envs/observation_tactical24.py`
- 행동: raw stick 4차원 [−1, 1], 10 Hz (60 Hz 시뮬레이션, action repeat 6)

## 작업 흐름

1. **환경 구축.** 주최측 시뮬레이터는 Windows DLL 기반이라 Linux에서 Wine + UDP 브리지로 구동했습니다.
2. **순수 RL 시도 (실패).** 보상 셰이핑과 커리큘럼으로 약 한 달을 썼습니다. 보상이 희소하고 계측 버그가 겹쳐 머지(정면 교전) 승리를 학습하지 못했습니다.
3. **스크립트 교사 → 모방학습.** 규칙 기반 교사(반전 → 요격 → 추적, `student/selfplay/autopilot_teacher.py`)를 BC와 DAgger 4라운드로 증류했습니다(`student/tools/bc_pretrain.py`).
4. **관측 확장.** 메모리 없는 정책은 교사의 래치 상태를 평균내 우유부단해졌습니다. 프레임 스택 관측(`tactical24_fv`)을 도입해 무승부 비율을 42%에서 29%로 줄였습니다.
5. **잔차 PPO.** 베이스를 동결하고 bounded residual head만 학습해 PPO 붕괴를 막았습니다(`residual_module.py`, `anchor_ppo_learner.py`). 이후 대회 고도(760 m)에 맞춰 `r11` 모델로 적응시켰습니다(`experiments/ours_ppo_r11_lowalt.yaml`).
6. **앵커 증류 → `p1d1`.** `r11`과 스크립트 브레이크를 합친 교사로 DAgger를 돌렸습니다. 브레이크는 신경망이 학습하지 못해 스로틀 거버너로 분리했습니다.
7. **터미널 인수 (핵심 성과).** 킬체인을 단계별로 분해해 보니(`student/eval/funnel_probe.py`) 병목은 "1° 조준 유지" 한 곳이었습니다. 조준을 신경망에서 떼어내 스크립트가 담당하게 하자 **킬률이 0.31%에서 14.8%로 올랐습니다**(264 에피소드, 사망 0).
8. **패리티 감사와 재훈련 (NO-GO).** 고도 드리프트와 상대 관측 디코드 버그를 찾아 고쳤고, 대회 고도 밴드에서 재훈련했지만 챔피언을 넘지 못했습니다.

| 킬체인 단계 (760 m) | 챔피언 NN | 스크립트 교사 |
|---|---|---|
| 분당 기회 창출 | 1.30 | 1.40 |
| 받은 피해 | 3.1% | 12.4% |
| 1° 조준콘 체류 시간 / 에피소드 | 0.13 s | 0.51 s |
| 창당 피해 | 1.4 pt | 8.0 pt |

앞 단계는 대등하거나 우세했고, 조준 유지(마지막 단계)만 끊겨 있었습니다.

## 교훈

- **대회 규칙부터 환경으로 옮길 것.** 고도, WEZ 스케줄, 행동 인터페이스를 늦게 맞춰 수 주를 잃었습니다.
- **보상은 점수와 같게 설계할 것.** 접근 기하에 보상을 주면 조준이 아니라 접근만 학습합니다(`student/my_reward.py`).
- **정밀 조준은 제어 법칙에 맡길 것.** 신경망에 조준을 가르치려 한 시도는 모두 실패했습니다.
- **계측을 먼저 검증할 것.** 캠페인 전체에서 발견한 계측·배관 결함이 24건입니다. 마지막 결함(배포 경로에서 스로틀 공간 변환 순서가 어긋나 브레이크가 0.15 대신 −0.70으로 나감)은 실제 대회 유닛에 들어가 있었습니다.

## 구조

```
src/dogfight/ai/
  terminal_takeover.py      터미널 조준 인수 (리드 추적 법칙)
  throttle_governor.py      접근 스로틀 거버너
  gcas.py                   예측형 지면 충돌 회피
  residual_module.py        동결 베이스 + bounded residual head
  anchor_ppo_learner.py     앵커 PPO (BC 정책 보존)
  asym_logstd.py            비대칭 log-std 클립
src/dogfight/envs/
  observation_tactical24.py tactical24 / tactical24_fv 관측
student/
  my_reward.py              보상 함수
  selfplay/                 스크립트 교사, 스크립트 상대
  tools/bc_pretrain.py      BC / DAgger 사전학습
  eval/funnel_probe.py      킬체인 퍼널 진단
experiments/
  ours_ppo_r11_lowalt.yaml  저고도(760 m) PPO 설정
```
