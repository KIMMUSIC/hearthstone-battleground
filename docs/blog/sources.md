# 첫 블로그 글 수정본 — 근거와 편집 메모

상태: **2026-09-06 게시 반영본**. 사용자의 게시·커밋·푸시·프로젝트 공개 전환 요청에 따라 반영했다.

[블로그 게시물](https://kimmusic.github.io/project/hearthstone-battleground-ai-1/)의 원문은 블로그 저장소 `_posts/2026-09-06-hearthstone-battleground-ai-1.md`다. 이 폴더의 [draft.md](draft.md)와 [index.html](index.html)은 검토 당시 형식을 보존한 미리보기이며 실제 Jekyll 결과는 아니다.

`python docs/blog/render_preview.py`로 Markdown을 HTML로 변환하고 네 그래프를 SVG/PNG로 내보낸다. 현재 PC에 설치된 Python-Markdown·Matplotlib과 맑은 고딕을 사용하며 프로젝트의 런타임 의존성은 바꾸지 않았다. 완성된 HTML과 SVG는 외부 폰트·JavaScript·분석 도구 없이 열린다.

## 편집 방향

- 독자: 전장을 알거나, 개인 AI 프로젝트가 실제로 어디까지 갔는지 궁금한 개발자.
- 읽은 글: [A2A 프로젝트](https://github.com/KIMMUSIC/kimmusic.github.io/blob/master/_posts/2025-10-07-a2a%20protocol.md), [뮤직 플레이어](https://github.com/KIMMUSIC/kimmusic.github.io/blob/master/_posts/2022-12-01-music%20player.md), [체스 통계](https://github.com/KIMMUSIC/kimmusic.github.io/blob/master/_posts/2025-12-20-chess-stats.md), AWS 실습·gRPC·BOJ 풀이 글.
- 제목은 기존 연재의 `만들기(1)` 형식, 소제목은 `[학습 환경]`, `[PPO 학습]`처럼 대상을 직접 명시한다.
- “만들어 보려고 한다”, “구성했다”, “확인해 보니”처럼 실제 작업을 설명하는 문장으로 바꾸고 포스터 문구·영문 배지·결심형 맺음을 제거했다.
- 수치와 코드, 짧은 설명이 이어지도록 구성했다. 학습 설정 → optimizer 업데이트 수 → 새 조건 평가 → 모방 학습의 상태 분포 문제 순서다.
- rollout/GAE/clipping, KL 로그의 계산 시점, NLL 기반 actor 학습과 critic 고정을 실제 코드·기록으로 설명한다. 하지 않은 학습이나 알고리즘 구현 경험을 추가하지 않는다.
- 동기 문장은 확인된 프로젝트 목표를 바탕으로 제안한 표현이다. 개인 플레이 경력·실제 레이팅 같은 미확인 경험은 넣지 않았다.
- 본문 GitHub 링크는 이미 push된 구현·실험 파일을 가리킨다. 게시 시 프로젝트 저장소를 공개로 전환한다.

## 레거시 한계

아래 파일은 읽기 전용 참고 프로젝트 `D:/HearthStoneAI`에 있다. 이 Git 저장소에는 레거시 전체를 중복 수록하지 않는다. 라인 번호는 2026-09-06에 확인한 로컬 사본 기준이다.

| 초안 주장 | 로컬 근거 | 해석의 범위 |
| --- | --- | --- |
| 기록상 보상 −15.3 → 49.0, 501,760스텝 | `WORK/WORK2/W2_training_results.md` | 당시 WORK 문서의 기록. 이번에 과거 학습을 재현한 것이 아니다. |
| seed가 서버 난수까지 전달되지 않음 | `grpc_battlegrounds_env.py:146`, `HearthstoneRL.Server/Data/CardDb.cs:16`, `HearthstoneRL.Server/Engine/CombatSimulator.cs:15` | Python reset은 seed를 받지만 RPC에는 Empty를 보내며 C#은 별도 Random을 생성한다. |
| 공유 상대 풀과 자기 매칭 가능성 | `HearthstoneRL.Server/Engine/GhostPool.cs:10`, `:33`; `HearthstoneRL.Server/Engine/BattlegroundsGame.cs:259`, `:262` | static pool에 자기 보드를 저장한 직후 상대를 뽑는다. 평가 조건이 실행 이력에 의존할 수 있다. |
| 카드별 효과 대신 공통 버프 | `HearthstoneRL.Server/Engine/BattlegroundsGame.cs:368`; `HearthstoneRL.Server/Engine/CombatSimulator.cs:283`, `:310` | BATTLECRY는 공통 종족 버프, DEATHRATTLE은 무작위 아군 +2/+2 처리. |
| 카드·매핑 버전 고정 필요 | `bg_card_db.py:18`, `:158`, `:161`; [Rebuild 설계](../DECISIONS.md) | latest 데이터와 정렬 기반 rl_id 재부여. Rebuild 문서는 원본 패치 ID가 불명이라고 명시한다. |

위 결함은 현재 코드에서 확인되는 설계 한계다. 각각이 과거 성능에 미친 인과적 기여도는 측정하지 않았다. “보상 해킹을 입증했다”, “학습은 전혀 이루어지지 않았다”라고 주장하지 않는다. 사용자의 역사적 Rebuild 결정 동기 전체를 대변하는 것도 아니다.

## 그래프와 수치

| 시각 자료 | 근거 | 분모 / 비교 조건 |
| --- | --- | --- |
| epoch별 optimizer 256 → 1024 | [007 결과](../EPOCH_COMPARISON_RESULT_007.md) | 단일 환경, 8192/32 rollout × epoch. |
| 교사 dev 약79% / 실제 방문 상태34–43% | [044 JSON](../../experiments/diagnostics/lobby044_visited_states.json) | 두 집단은 다른 상태. dev 2236행, 방문 상태 1373/1436/1529행. renderer가 정확도를 JSON에서 읽는다. |
| 작은 환경 생존율, 새 seed: 0 → 71.67% | [008 결과](../HOLDOUT_RESULT_008.md), [보존 집계](evidence/holdout-008-audit.json) | 조건당 3 학습 seed × 3 정책 seed × 20 게임 seed = 180회. 휴리스틱 20회, 생존율 100%. |
| 상대 변경: 0 → 73.33% | 같은 008 결과·집계 | 기존 상대 조건과 같은 게임 seed 301–320을 사용. 고유 게임 seed를 40개로 세지 않는다. |
| 스텝 증가만으로 안정적 개선 없음 | [006 결과](../LEARNING_CURVE_RESULT_006.md) | 512/2048/8192스텝 × 학습 seed 7/17/27. 최고 모델만 선택하지 않는다. |
| epoch 1 → 4 비교 | [007 결과](../EPOCH_COMPARISON_RESULT_007.md) | 8192 환경 스텝 고정, optimizer 업데이트 256 → 1024. 동일 계산량 비교는 아니다. |
| 045 평균 순위 6.90 → 7.00 | [045 결과](../daily/2026-10-16.md), [선별 집계](evidence/lobby-045-audit.json) | 최종 3 모델 seed 각각 20경기, 조건당 60경기. 반복 개발 평가 세트. 각 점의 분모는 20. |
| 같은 평가의 휴리스틱 순위 4.25 | 위 045 집계의 `baselines.heuristic.mean_rank` | 휴리스틱 20경기. 학습 모델 3seed 평균과 분모를 구분한다. |
| 출처별 정확도 상충 2/3 seed | [046 결과](../daily/2026-10-17.md), [진단 JSON](../../experiments/diagnostics/lobby046_source_diagnosis.json) | 학습 데이터 정확도이며 개발 정확도·경기 성능이 아니다. 원인 확정이 아니다. |
| 다음 48/16 sampling 및 채택 조건 | [047 명세](../../experiments/LOBBY_047_SOURCE_MIX.md) | 계획만 완료. 기본/추가 상태 비중과 batch 간 비중 분산을 함께 바꾼다. |

008은 당시 새로운 평가 seed를 사용한 실험이다. 그 후 결과가 개발 과정에 알려졌으므로 현재의 최종 미사용 holdout이라고 주장하지 않는다. 생존율은 제한된 고정 상대 환경의 에피소드 생존이며 실제 전장 승률·MMR·8인 top4가 아니다. 모든 그래프는 기술 통계이며 통계적 유의성이나 전체 게임 일반화 주장을 하지 않는다.

045 집계는 2.46 MB 전체 audit 중 본문 근거에 필요한 필드만 보존했다. 원본 경로·SHA256은 [provenance.json](evidence/provenance.json)에 있다. 원본 모델, 개별 trajectory, 전체 audit는 로컬 `runs/`와 `handoff/`에 보존하며 Git에서 제외한다. 이 작은 스냅샷만으로 과거 학습을 처음부터 재현할 수 있다고 주장하지 않는다.

## 날짜와 현재 상태

최신 완료 실험은 046, 다음은 047이다. 문서 `2026-10-16`, `2026-10-17` 등의 날짜는 작업 순서 라벨이다. 해당 문서에 실제 수행일 **2026-09-06**이 명시되어 있다. 초안에서는 실험 번호를 사용했다. 후속 학습이나 카드 확장을 이번 블로그 작업에서 시작하지 않았다.

## 검증

그래프는 008 집계의 9개 stochastic cell, 044의 두 상태 집단 정확도, 045의 seed별 순위를 읽어 만든다. 링크 경로·그림 로드·모바일 배치·펼침 설명을 검증한다. 이번 변경은 블로그 초안에만 한정되므로 전체 학습 테스트를 재실행하지 않았다. [소스 스냅샷](../SOURCE_SNAPSHOT.md)의 383 테스트 통과는 이전 커밋 시점 기록이다.

## 게시 자료

블로그에는 기존 project 카테고리, header.teaser, 날짜와 제목 규칙을 적용했다. 네 그래프는 PNG로 포함했고 별도 미리보기의 CSS 의존 도식은 Markdown 표로 옮겼다. 프로젝트 README는 소개·목표·현재 범위와 문서 링크로 줄였다.

썸네일은 Blizzard Entertainment의 [Battlegrounds Revamp Coming Tomorrow!](https://hearthstone.blizzard.com/en-us/news/23714527) 공식 게시물의 대표 이미지다. [원본 이미지](https://bnetcmsus-a.akamaihd.net/cms/blog_header/oj/OJF7LE4UGOVJ1629918279214.jpg)를 수정 없이 블로그 asset으로 저장하고 본문에 출처와 저작권자를 표시했다. 직접 구현한 게임 화면이나 AI 성능 결과 이미지가 아니다.
