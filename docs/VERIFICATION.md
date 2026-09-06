> 역사적 기록: 2026-09-06부터 실행은 [로컬 WSL 계약](LOCAL_EXECUTION.md)으로 전환했다. 아래 당시 상태는 현행 실행 지침이 아니다.

# 로컬 구현 검증 — 2026-09-05

## 완료한 범위

`archival-subset-v1`(카드 4장, 최대 2티어)의 엔진, Gym 관측·마스크, 무작위·규칙 기준 정책, 고정 평가, 짧은 CPU PPO 학습·저장·재개를 구현했다. 단계 0–3의 로컬 실행 경로에 해당한다. 특정 실제 패치의 완전한 전장 규칙 검증은 아니다.

## 실제 확인 결과

| 검사 | 결과 |
| --- | --- |
| `python -B -m pytest -q` | **49 passed**, 16.00초 |
| `python -m ruff check src tests scripts` | All checks passed |
| Python 소스·테스트·스크립트 AST 파싱 | 통과 |
| `python -B scripts/bgai.py smoke --seed 42` | 3번째 전투 후 horizon 종료 |
| 카드·공격 순서·도발·보호막·불변성 | 회귀 테스트 통과 |
| 손패·트리플·발견·액션 계약·seed·종료 | 회귀 테스트 통과 |
| 관측 용량·음수 HP·위치·황금 여부 | 테스트 통과 |
| 고정 평가 재현·입력 불변성 | 테스트 통과 |
| 별도 프로세스·다른 cwd에서 비기본 horizon 모델 평가 | 통합 테스트 통과 |
| 체크포인트 불일치·실패·중단·중복 잠금 | 테스트 통과 |

동작 검증에는 기본 horizon 외에도 2턴/행동 4회 설정에서 32스텝 학습→별도 프로세스 평가→64스텝 재개가 포함된다. 모델 SHA가 평가 결과에 기록되는지와 옵티마이저 업데이트 카운터 증가도 확인한다.

## 보존된 최종 소스의 실행 결과

소스 SHA-256:

`5ce5003ef0c9329e7b54d9898aeda4376d80e8d853c6c227550413a9d325d142`

- [최종 smoke 상태](../runs/verified-smoke-001/status.json): 0→64스텝, PyTorch CPU 스레드 1, 학습 함수 경과 2.578초
- [재개 상태](../runs/verified-resume-001/status.json): 64→128스텝, 같은 설정, 학습 함수 경과 3.328초
- [학습 환경](../runs/verified-smoke-001/environment.json), [학습 이벤트 로그](../runs/verified-smoke-001/train.log), [자원 지표](../runs/verified-smoke-001/metrics.jsonl)
- [모델 메타데이터](../runs/verified-smoke-001/checkpoint-000000000064-0002/metadata.json)

경과 시간은 Python 시작·import 전체 시간이 아니며 성능 벤치마크로 일반화하지 않는다. 실험 결과는 `.gitignore`의 `runs/` 아래에 있으므로 소스만 전달할 때 자동 포함되지 않는다. 초기 `integration-*` 실행은 후속 코드 변경 전 검증 기록이며 현재 서명과 일치하는 최종 결과는 `verified-*`다.

## 작은 고정 평가 표본

seed 101–105, 5게임, 기본 8턴·행동 24회. 상대 세트 해시:

`84f48a6b713165ec3ddfc7caeb6136fe0f4f5de36b01f55f383c4607e09a7fa2`

| 정책 | 전투 승률 | horizon까지 생존 | 평균 총 보상 |
| --- | --- | --- | --- |
| [무작위](../runs/verified-random.json) | 0% (31전투) | 0/5 | -5.6 |
| [단순 규칙](../runs/verified-heuristic.json) | 67.5% (40전투) | 5/5 | 5.4 |
| [64스텝 smoke 모델](../runs/verified-smoke-001/evaluation.json) | 17.5% (40전투) | 4/5 | -2.2 |

이는 파일 연결과 평가 지표 검증용 소표본이다. 학습 전 동일 초기 모델과의 비교, 여러 학습 seed, 통계적 불확실성 검증을 하지 않았으므로 학습 효과나 실제 전장 실력을 입증하지 않는다. 현재 smoke 모델은 규칙 정책보다 낮은 전투 승률이다.

## 코드 검토 후 반영

- 비기본 `max_turns`·`max_actions` 학습 모델을 CLI로 평가할 수 있도록 옵션과 통합 테스트 추가
- 평가 대상 모델의 경로·SHA-256 기록
- 전체 소스 checkout/editable install만 지원한다는 배포 범위 명시. 일반 wheel 자산 패키징은 미구현

## 환경과 검증 한계

- Windows 10.0.26200, Python 3.11.5. 초기 검사에서 논리 CPU 12개를 확인했다.
- 사용한 직접 의존성: NumPy 2.4.2, Gymnasium 1.2.3, Torch 2.10.0, SB3/sb3-contrib 2.7.1. 검사 도구: pytest 9.0.2, Ruff 0.15.12.
- 이번 작업에서 새 패키지를 설치하지 않았다. Windows 의존성 버전 목록을 `requirements-windows.lock`에 남겼다. 깨끗한 가상환경 재설치·Linux 실행·wheel 배포는 검증하지 않았다.
- mypy 등 별도 타입 검사기는 설치되지 않아 실행하지 않았다. Ruff 정적 검사, 파싱, 실행 테스트로 검증했다.
- 재개는 모델·옵티마이저·누적 스텝을 복구하며, 에피소드·전체 RNG 상태를 복구하지 않는다.
- 시간 제한은 단계 사이에서 검사하는 협력적 제한이다. 메모리 강제 상한·OS 프로세스 감독·원격 전달·YAML 실행기는 후속 운영 작업이다.
- Grok Bot의 장비 접근과 실제 학습 운영은 아직 미검증이다. Bot에 작업을 전달하거나 자동화를 등록하지 않았다.
- 기존 `D:\HearthStoneAI` 소스와 모델은 수정하지 않았다.
