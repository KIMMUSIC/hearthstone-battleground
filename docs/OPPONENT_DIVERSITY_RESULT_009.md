# Opponent Diversity Result 009

상태: **완료 — diverse-v1 후보 기각**. 실행 위치는 `runs/diversity-009-local-r1/`이며, Grok Bot이 아닌 로컬 WSL Ubuntu / CPython 3.11.15 / CPU에서 수행했다.

## 결론

`diverse-v1`은 연구 기준 후보로 채택하지 않는다. 사전 기준은 평균 보상 delta `>= 0.25`, 양성 학습 seed 2개 이상, stochastic 생존율 하락 5%p 이하였지만, 실제로는 모든 seed에서 `diverse-v1`이 `fixed-v1`보다 낮았다.

| 항목 | 결과 |
| --- | --- |
| campaign | `opponent-diversity-009` |
| run ID | `diversity-009-local-r1` |
| source SHA256 | `81b0dae34dfbac6bf875d1ebcf2647f34e68817b1b0987bc63685801e614cd90` |
| 감사 상태 | `passed` |
| 오류 | `[]` |
| 평균 delta, diverse-fixed | `-1.4194444444444443` |
| 양성 seed | `0/3` |
| stochastic 생존율 delta | `-0.17500000000000004` |
| candidate_keep | `false` |

## 실행 범위

학습은 두 상대 조건과 세 학습 seed를 같은 순서로 수행했다.

| mode | seed | steps | optimizer updates |
| --- | ---: | ---: | ---: |
| fixed-v1 | 7 | 8,192 | 1,024 |
| diverse-v1 | 7 | 8,192 | 1,024 |
| fixed-v1 | 17 | 8,192 | 1,024 |
| diverse-v1 | 17 | 8,192 | 1,024 |
| fixed-v1 | 27 | 8,192 | 1,024 |
| diverse-v1 | 27 | 8,192 | 1,024 |

평가는 reference와 shifted cohort에 대해 각각 50개 보고서를 만들었다. 각 cohort는 초기/최종 모델, 두 mode, 세 학습 seed, argmax와 policy seed 11/22/33 sampling, 그리고 random/heuristic 기준선을 포함한다. 총 평가 게임 수는 2,000이다.

## 감사 근거

| 파일 | SHA256 |
| --- | --- |
| `runs/diversity-009-local-r1/audit.json` | `e8cf87c125bb108b8c3df4f028853ae52eb14ba125bd8bebcd2a0757d9ccd504` |
| `runs/diversity-009-local-r1/outputs/train/summary.json` | `1053afc4b54c4f6ac04b94046d7ff636734c6eec70e35e5053cc1734e6e1e10f` |
| `runs/diversity-009-local-r1/outputs/eval-reference/summary.json` | `0302e1586d1ee3ae67fb7de5c5bb1bbcc11266ce758e3f308b8d7d258e7e071c` |
| `runs/diversity-009-local-r1/outputs/eval-shifted/summary.json` | `2fdd272b2794829995703c9f1f2d3a245995240bcd330cb98c3404613396aed9` |
| `runs/diversity-009-local-r1/local-verification.json` | `7f7c8277ca348fb81436c23a01087771d69911025e92bb24091107ceea8f4140` |

환경/호환성:

| 항목 | 값 |
| --- | --- |
| Python | `3.11.15` |
| NumPy | `2.4.2` |
| Torch | `2.10.0+cpu` |
| Stable-Baselines3 | `2.7.1` |
| sb3-contrib | `2.7.1` |
| action version | `discrete-37-v1` |
| observation version | `onehot-slots-v1` |
| cards SHA256 | `8e02d494165c4d5be64e4f05728346801b0c6e5f5e85488285cb71061a11bf0d` |
| mapping SHA256 | `6d80f62da8b29f16b0cccad1ff36af09b6a3509b2f10585ab6e3acb3a2b01785` |
| rules SHA256 | `a2ddcd9749dbf3cdeee6d2de4a72963389bbc59c08c9bee11c63ec134ef3ffb7` |

## 성능 결과

Primary decision은 최종 모델의 stochastic 평가만 사용해 seed별로 reference와 shifted를 합산했다.

| 학습 seed | fixed 평균 보상 | diverse 평균 보상 | delta |
| --- | ---: | ---: | ---: |
| 7 | -3.0083 | -4.5667 | -1.5583 |
| 17 | -2.3000 | -3.7000 | -1.4000 |
| 27 | -1.7583 | -3.0583 | -1.3000 |

최종 모델 전체를 보면 diverse는 두 cohort 모두에서 reward와 survival이 낮았다.

| cohort | mode | 보고서 | 평균 보상 | 생존율 | combat win rate |
| --- | --- | ---: | ---: | ---: | ---: |
| reference | fixed-v1 | 12 | -2.3250 | 0.6875 | 0.2111 |
| reference | diverse-v1 | 12 | -3.7583 | 0.4500 | 0.1376 |
| shifted | fixed-v1 | 12 | -1.7167 | 0.7417 | 0.2448 |
| shifted | diverse-v1 | 12 | -3.1583 | 0.5500 | 0.1812 |

Stochastic 선택만 보면 리롤과 행동 수 증가가 같이 나타난다.

| cohort | mode | 평균 행동 수 | 평균 강제 턴 | 평균 리롤 | 평균 스왑 |
| --- | --- | ---: | ---: | ---: | ---: |
| reference | fixed-v1 | 112.13 | 2.63 | 2.83 | 11.61 |
| reference | diverse-v1 | 131.13 | 3.68 | 7.43 | 13.19 |
| shifted | fixed-v1 | 112.72 | 2.66 | 2.82 | 11.66 |
| shifted | diverse-v1 | 132.06 | 3.70 | 7.54 | 13.33 |

## 해석

009는 상대 다양화 구현의 smoke와 감사에는 성공했지만, 성능 개선 실험으로는 실패했다. 현재 증거는 전투 규칙 결함보다 정책의 action 선택 문제가 우선이라는 쪽에 더 가깝다.

근거:

- `selection_probe.py`는 모델 확률이 action mask를 위반하면 실패한다.
- `audit_diversity_009.py`는 trace의 action mask, 확률 합, illegal action 확률, reward 합을 재검산한다.
- 감사는 오류 없이 통과했다.
- 나쁜 pair는 forced end turn, 리롤 증가, 스왑 부재 또는 과다 스왑처럼 합법 action 안의 선택 품질 문제로 보인다.

따라서 `fixed-v1`을 다음 기준선으로 유지하고, `diverse-v1`은 별도 후보로 보존하되 승격하지 않는다. 후속 조사는 `experiments/FOLLOWUP_010.md`의 학습 없는 trace 진단으로 제한한다.

## 다음 작업

1. `FOLLOWUP_010` 재생 완료: 960게임·123519행동의 상태·mask·reward·종료가 고정 엔진과 일치했다.
2. 구현 결함이 확인되면 카드 확장보다 먼저 실패 테스트와 수정으로 닫는다.
3. 구현 결함이 아니면 정책/보상/행동 예산 가설을 다음 소규모 실험 명세로 작성한다.
4. 카드 효과 확장은 골렘 단순 죽메 소환을 첫 후보로 두고, 뱃사람 즉시 공격은 보류한다.
