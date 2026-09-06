# Follow-up 010 — 정책 행동 낭비 진단

상태: **완료 — 최종 정책960게임의123519행동 재생 일치**. 이 follow-up은 009 결과를 설명하기 위한 학습 없는 진단이다. 새 학습, 새 대규모 평가, 카드 확장은 포함하지 않는다.

## 배경

`opponent-diversity-009`의 로컬 run `runs/diversity-009-local-r1`은 감사 통과 상태로 끝났지만, `diverse-v1`은 사전 채택 기준을 통과하지 못했다.

핵심 수치:

- 평균 delta, diverse-fixed: `-1.4194444444444443`
- 양성 학습 seed: `0/3`
- stochastic 생존율 delta: `-0.17500000000000004`
- 후보 채택: `false`

9/11 trace 확인에서는 불법 action이나 action mask 위반보다, 합법 action 안에서 리롤·스왑·긴 턴이 늘어나는 정책 문제가 먼저 보인다.

## 목표

하나의 질문에 답한다.

> 현재 실패는 엔진 규칙 결함인가, 아니면 정책/보상/행동 예산이 만드는 합법적이지만 나쁜 선택 패턴인가?

완료 시점에는 다음 중 하나로 판정한다.

- **구현 결함:** 특정 trace가 규칙 계약과 충돌한다. 이 경우 먼저 실패하는 회귀 테스트를 작성하고 수정한다.
- **정책 결함:** trace와 코드가 규칙 계약 안에서 일치하지만 정책 선택이 나쁘다. 이 경우 reward/action penalty/termination shaping 같은 다음 실험 가설을 문서화한다.
- **증거 부족:** 추가 로그 필드가 필요하다. 이 경우 필요한 필드만 명세하고 학습은 시작하지 않는다.

## 입력

읽기 전용 입력:

- `runs/diversity-009-local-r1/audit.json`
- `runs/diversity-009-local-r1/outputs/eval-reference/*.json`
- `runs/diversity-009-local-r1/outputs/eval-shifted/*.json`
- `runs/diversity-009-local-r1/src/hearthstone_ai/actions.py`
- `runs/diversity-009-local-r1/src/hearthstone_ai/game.py`
- `runs/diversity-009-local-r1/scripts/selection_probe.py`
- `runs/diversity-009-local-r1/scripts/audit_diversity_009.py`

원본 run 디렉터리의 학습 출력과 평가 JSON은 보존한다. 분석 산출물이 필요하면 새 파일을 `docs/daily/` 또는 별도 분석 문서에 기록하고, run 원본의 JSON을 고치지 않는다.

## 진단 절차

1. 최종 모델의 fixed/diverse pair를 같은 cohort, training seed, selection, policy seed로 묶는다.
2. 각 pair에서 reward delta, survival delta, forced end turn delta, action count delta, reroll delta, swap delta를 계산한다.
3. delta가 가장 나쁜 trace를 골라 action sequence를 action 이름으로 decode한다.
4. 각 step의 action이 `legal_actions`와 `action_mask`에 포함되는지 재확인한다.
5. forced end turn이 발생한 step에서 `actions_remaining`, `gold`, `frozen`, 이전 action 종류를 확인한다.
6. 코드의 `Game.step`, `Game.legal_actions`, `_end_turn`, `selection_probe.probe`가 trace 해석과 모순되는지 확인한다.

이 절차는 재현 분석이며 학습을 시작하지 않는다.

## 우선 판정 기준

구현 결함으로 판정하는 조건:

- trace의 action이 mask에서 illegal인데 실행됐다.
- 기록된 행동 전 `actions_remaining`과 실제 전이가 다르다. trace는 행동 전 값이므로 보통 1에서 행동을 소모한 뒤 강제 종료가 발생한다. 발견 선택 지연도 실제 전이로 검증한다.
- pending discover, hand/board 용량, swap 횟수, freeze 상태가 `game.py` 계약과 충돌한다.
- 같은 seed와 같은 action sequence가 재현되지 않는다.

정책 결함으로 판정하는 조건:

- 모든 action은 legal이고 audit도 확률/mask를 통과한다.
- 실패 pair에서 forced end turn, 리롤, 무의미한 스왑, early end 중 하나가 반복적으로 높다.
- `Game.step`의 상태 전이가 trace와 일치한다.

## 현재 후보 결론

현재까지의 우선 후보는 **정책 결함**이다. `selection_probe.py`는 mask 위반을 오류로 처리하고, `audit_diversity_009.py`는 trace별 확률 합과 illegal 확률을 재검산한다. 감사가 통과했으므로 첫 수정 대상은 action registry나 mask가 아니라, 정책이 합법 action 중 나쁜 선택을 고르는 원인이다.

관찰된 나쁜 pair:

| pair | fixed reward | diverse reward | 핵심 차이 |
| --- | ---: | ---: | --- |
| shifted seed27 argmax | 3.6 | -6.0 | diverse forced 6, 스왑 0, 생존 0.0 |
| reference seed27 argmax | 2.7 | -6.0 | 같은 패턴 |
| reference seed7 sample22 | -3.0 | -5.5 | diverse 행동 수 168.7, 리롤 13.4, forced 6.5 |

## 제외 범위

- 새 PPO 학습을 실행하지 않는다.
- 009 결과의 판정 기준을 소급 변경하지 않는다.
- `diverse-v1`을 기본 설정으로 승격하지 않는다.
- 골렘이나 뱃사람 효과를 이 follow-up 안에서 구현하지 않는다.

## 후속 결정

진단 결과가 정책 결함이면 다음 실험은 학습 전 명세만 작성한다. 후보는 action budget 낭비를 줄이는 보상/종료/마스크 조정 중 하나여야 하며, 상대 다양화와 카드 확장을 동시에 바꾸지 않는다.

진단 결과가 구현 결함이면 카드 확장보다 먼저 테스트와 수정으로 닫는다. 수정 후에는 기존 009 모델을 새 결과처럼 재해석하지 않고, 새 run ID로 필요한 최소 smoke와 평가만 별도 명세한다.

## 실행 결과

`python -B scripts/replay_trace_010.py --snapshot runs/diversity-009-local-r1 --output docs/daily/2026-09-11-replay.json` 종료0. 최종모델48보고서·960게임·123519행동을 고정 실행본 엔진에 재생했다. 매 행동 전 turn/gold/frozen/actions_remaining/전체mask, 행동 후 reward/forced-end와 최종HP·종료·보상합이 모두 원본과 일치한다. 학습과 모델추론은 추가하지 않았다.

판정: 검사 범위에서 기록과 엔진 구현의 불일치는 발견하지 못했다. 합법적 반복 행동의 선택 품질 문제라는 가설을 유지한다. 라이브 게임 규칙의 정확성을 이 재생만으로 증명하지 않는다. 다음 연구 가설은 반복 동결의 행동 비용을 명시하면 행동 낭비가 줄어드는지로 하나만 좁힌다. 보상 변경 학습은 이번 범위에서 실행하지 않았으며, 새 실험에 앞서 보상식·예산·평가 기준을 별도 고정해야 한다.
