# 041 — 행동 모방을 통한 신경망 학습 가능성 진단

040에서11,435관측의 heuristic 행동 복원이 전부 일치했다. 관측 정보가 충분한 조사 범위에서 동일32×32 정책망이 좋은 행동 예시를 배울 수 있는지 별도 진단한다. PPO 성능 채택 실험이나 실제 전장 성능 주장은 아니다.

## 데이터와 모델

고정 heuristic7 로비에서 학습 seed0..99의 learner heuristic trajectory를 모방 데이터로 만든다. 개발 seed20001..20020의 teacher trajectory는 분리 저장하고 optimizer에 전달하지 않는다. fixed-v1 관측, 같은 카드·보상·액션 계약, 공개정보만 사용한다. 각 파일은 obs__<field>, actions, seeds의 숫자 NPZ이며 pickle을 허용하지 않는다. 해시·표본수·seed·합법 마스크·완주/보존을 기록한다.

seed7·17·27의 새로운 MaskablePPO 형식32×32 정책을 만든다. PPO rollout/보상 최적화는 수행하지 않는다. teacher action의 masked negative log likelihood를 Adam learning_rate0.0003, batch64, 정확히1024회로 최적화한다. 샘플은 각 모델 seed의 로컬 RNG로 학습 split에서 균등 복원추출한다. clip_grad_norm0.5, CPU1, 각60초 상한. critic 가중치는 업데이트하지 않으며 초기/최종 비교로 확인한다.

환경 num_timesteps와 SB3 _n_updates는0으로 보존한다. metadata에 training_method=behavior-cloning-diagnostic-v1, 실제 supervised optimizer_steps와 rows_seen을 기록한다. 이를8,192스텝 PPO 학습으로 표시하지 않는다. 현재 환경 호환 검사와 원본 모델 불변 규칙은 유지하며 기존 모델에 이어 학습하지 않는다.

## 평가와 사전 판정

개발 teacher trajectory의 masked top1 일치율, negative log likelihood, 행동별 표본수/정확도를 초기/최종 모두 보고한다. 추가로 각 모델을 자기 행동으로20개 공통 heuristic7 개발경기에 실행한다. 초기/최종3모델×20=120경기, heuristic/random 각20=40, 총160경기 및 최종 첫2seed 추가재현. 학습 데이터를 잘 맞추는 것과 자기 행동으로 경기를 잘 하는 것을 구분한다.

행동 모방 가설: 최종 개발 일치율≥80% 및 초기보다 개선인 seed가2/3이상. 경기 전이 가설: 평균 초기→최종 순위 개선≥1.0, 개선 seed≥2/3, 최종 평균 상위4위 비율≥10%, 제한중단0. 두 가설을 별도 판정한다. 모든 seed와 행동별 실패를 공개하고 실패를 재학습 증액으로 덮지 않는다.

두 기준이 모두 지지되면 모방 초기화 뒤 PPO를 연결하는 별도 비교를 명세한다. 일치율만 높고 경기 성능이 낮으면 학습 데이터와 실제 행동 경로 차이를 조사한다. 일치율도 낮으면 모델/데이터/행동 분류 오류를 먼저 진단한다. 카드 확대와 대규모 예산 증액은 자동 진행하지 않는다.

## 실행 및 감사

dataset1단계+train3단계+evaluate3단계+baseline1단계=8개 감독 단계. 각180초, 샘플링RSS2GiB, 출력+반환1GiB, 학습당60초. 고유run·동결입력·빈출력·자동재시도 없음. 학습예제 노출은모델당65,536개이며 환경 스텝과 구분한다.

완료 증거: split분리와 데이터해시/teacherlabel 재생성 감사, 실제actor Adamstep1024·critic불변·정확한0 PPO카운터, 개발분류지표 재계산,160경기 모든행동 재현,입력/model/ZIP해시 및 자원 감사. 예약검증22001..22100과 기존 최종seed는 미사용이다.
