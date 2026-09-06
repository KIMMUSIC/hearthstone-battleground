# HearthStoneAI Rebuild

정확한 소규모 하스스톤 전장 시뮬레이터와 재현 가능한 강화학습·평가 환경을 만드는 새 프로젝트다.

2026-09-06 첫 Git 스냅샷의 포함 범위와 fresh 검증(WSL 383 tests passed)은 [소스 스냅샷 기록](docs/SOURCE_SNAPSHOT.md)에 있다. 아래 일일 기록의 10월 날짜는 작업 순서 라벨이며 실제 수행일과 구분한다. [첫 블로그 초안](docs/blog/index.html)은 미게시 검토용 HTML이다.

현재는 상점 카드 7종·소환 전용 토큰 1종·2티어의 제한된 엔진, 고정 평가, CPU PPO 학습·저장·재개를 구현했다. 일반 골렘의 사망 소환을 지원하며 골렘 황금·트리플과 즉시공격 죽메는 제외한다. 최신 전장 패치를 재현하는 완성 엔진은 아니다. 정확한 범위는 [규칙 계약](docs/RULES.md), 최신 검증은 [일일 결과와 실행 계획](docs/DAILY_GOALS_2026-09.md), 초기 이력은 [검증 기록](docs/VERIFICATION.md)을 참고한다.

제한된 8인 로비와 공유 유한 풀도 구현·검증했다. 100경기 자연 종료, 전체 행동 재현, 276테스트와 Ruff 통과. 8인 PPO 학습 어댑터도 smoke 수준으로 연결했고 `lobby-032-local-r1`에서 8,192스텝과 60개 개발 평가 에피소드 audit가 통과했다. 다만 최종 모델은 heuristic 기준선보다 낮아 성능 개선 증거는 없다. [지원 규칙과 검증](docs/LOBBY_VERIFICATION.md), [10/2 결과](docs/daily/2026-10-02.md)를 참고한다.

후속039 rollout 비교까지 완료했다. 평균 순위 개선0.0833은 채택 기준에 미달했다. 040 관측 정보 진단에서 11,435상태의 행동 복원이 모두 일치했다. 041 모방 학습도 완료했으나 일치율과 경기 성능은 사전 기준 미달이다. 042에서 행동별 학습 편차와 자기 방문 상태의 일치율 저하를 확인했다. 043 균형 샘플링은 분류만 개선해 미채택이다. 044에서 실제 방문 상태의 큰 일치율 격차를 확인했다. 045 방문 상태 추가도 평균순위 악화와정확도미달로미채택이다. 046에서2/3seed의데이터출처간정확도상충을확인했다. 다음은047 기본/추가48/16샘플quota 비교다. [10/17 결과](docs/daily/2026-10-17.md)와 [현재 후속 계획](docs/PLAN_AFTER_2026-10-02.md)을 참고한다.

## 시작할 때 읽을 문서

1. [AGENTS.md](AGENTS.md): 개발 에이전트의 작업 원칙과 책임
2. [개발 단계와 완료 기준](docs/DEVELOPMENT.md): 구현 순서와 검증 기준
3. [로컬 WSL 실행 계약](docs/LOCAL_EXECUTION.md): 현재 로컬 실행 경로, venv, run ID, 자원 한도
4. [Linux 실행 감독기](docs/SUPERVISOR.md): 로컬 WSL 실행, 자원 감시, 종료·오류 전파
5. [실험 명세 템플릿](experiments/EXPERIMENT_TEMPLATE.yaml): 실행 전에 확정할 입력과 제한
6. [2026년 9월 날짜별 계획](docs/PLAN_2026-09.md): 9월 6일부터의 개발·실험 일정과 단계별 완료 기준
7. [하루 단위 goal 실행 계획](docs/DAILY_GOALS_2026-09.md): 28개 일일 목표, goal 입력용 문구, 검증·재개 기준

## 목표와 범위

- 첫 목표는 제한된 카드 풀에서 규칙이 정확하고 결과가 재현되는 학습 환경이다.
- 실제 게임의 기준 패치와 카드 목록을 명시한다. 임의로 단순화한 규칙은 별도 규격으로 기록한다.
- 현재 단계의 성과를 실제 전장 승률이나 MMR로 표현하지 않는다.
- 실제 클라이언트 조작, 모든 카드 구현, 대규모 탐색 알고리즘은 제외한다. 8인 대전은 명시한 부분 규격에서 지원한다.
- 기존 프로젝트 `D:\HearthStoneAI`는 읽기 전용 참고 자료다. 기존 코드와 모델은 검증 없이 복사하거나 이어서 학습하지 않는다.

## 역할

| 담당 | 책임 |
| --- | --- |
| Codex | 엔진·학습 코드·테스트 구현, 실험 설계, 로컬 실행 운영, 자원 감시, 체크포인트·로그 보존, 고정 평가와 결과 분석 |
| 사용자 | 목표 범위, 사용할 실행 장비, 자원·비용 한도 결정 |

## 첫 개발 작업

현재 구현의 설계 판단은 [설계 결정](docs/DECISIONS.md)에 기록했다. 현재 실행 경로는 사용자의 로컬 PC이며, 권장 방식은 WSL Ubuntu에서 전용 Python 3.11 가상환경과 Linux 감독기를 사용하는 것이다. 장시간 학습과 실제 패치 카드 확장은 별도의 검증 단계다.

## 실행

프로젝트 루트에서 Python 3.11로 실행한다. 표준 라이브러리만으로 기본 엔진 동작을 확인할 수 있다.

```powershell
python -B scripts/bgai.py doctor
python -B scripts/bgai.py smoke --seed 42
```

학습·평가에는 `pyproject.toml`의 rl 의존성이 필요하다. **현재 지원 배포 방식은 전체 소스 checkout**이다. 일반 wheel 설치는 지원하지 않는다. 이 폴더 전체를 복사하거나 editable install을 사용해야 `data/`, `configs/`, `docs/`가 유지된다.

새 Windows 환경의 설치 예시(현재 PC에는 필요한 패키지가 이미 있어 이번 구현에서 추가 설치하지 않았다):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-windows.lock
.\.venv\Scripts\python -m pip install --no-build-isolation --no-deps -e .
```

로컬 운영은 WSL Ubuntu를 기본으로 한다. Windows 전역 Python은 빠른 확인에 사용할 수 있지만, 감독기 기반 학습·평가는 WSL 안에서 Python 3.11 전용 venv를 만들고 의존성과 짧은 실행을 다시 검증한 뒤 수행한다.

```powershell
python -B -m pytest -q
python -m ruff check src tests scripts
python -B scripts/bgai.py evaluate --policy heuristic --output runs/heuristic-new.json
python -B scripts/bgai.py train --config configs/smoke_train.json --output runs/smoke-new
```

학습은 기본 64스텝, CPU 1스레드, 협력적 시간 제한 60초다. 출력 폴더는 비어 있어야 한다. 결과의 `checkpoint` 값을 이용해 다음 명령을 실행한다.

```powershell
python -B scripts/bgai.py evaluate --policy model --checkpoint runs/smoke-new/<checkpoint> --output runs/smoke-new/evaluation.json
python -B scripts/bgai.py train --config configs/smoke_train.json --resume runs/smoke-new/<checkpoint> --output runs/resume-new
```

`<checkpoint>`는 실제 출력된 디렉터리 이름으로 바꾼다. `max_turns`·`max_actions`를 바꿔 학습했다면 평가에도 `--max-turns`·`--max-actions`로 같은 값을 전달한다. 재개는 이전 누적 스텝에 설정의 `max_steps`를 추가하며 에피소드와 RNG 궤적은 새로 시작한다. 소스·규칙·카드 매핑·환경 서명이 달라진 모델은 로드하지 않는다.

장비 한도·메모리 감시를 갖춘 전체 운영용 YAML 실행기는 아직 구현하지 않았다. 현재 `train` 명령은 `configs/smoke_train.json` 형식의 제한된 JSON 설정을 받는다. `experiments/EXPERIMENT_TEMPLATE.yaml`은 후속 운영 규약 템플릿이다.

과거 원격 실행에서 64→128스텝 재개와 네 정책의 20게임 평가까지 확인했다. 학습 성능 향상은 입증되지 않았고, 운영 스크립트의 종료·오류 전파 결함이 발견됐다. 결과는 `docs/GROK_HANDOFF_001.md`, 다음 검증 계획은 `experiments/NEXT_RUN_003.md`에 기록했다.

후속 구현은 초기 checkpoint(`checkpoint-000000000000`)와 `evaluate --trace` 행동 기록을 지원한다. `scripts/supervise.py`는 Linux에서 캠페인 제한·프로세스 그룹 종료·실패 전파를 담당하며, `scripts/bgai_threaded.py`는 실제 RL 프로세스의 Torch 스레드를 설정한다. 사용법과 샘플링 RSS의 한계는 `docs/SUPERVISOR.md`를 따른다. 소스 변경 전 checkpoint는 새 실행본에서 호환성 검사를 우회해 로드하지 않는다.

새 실행본의 원격 테스트 60개와 미학습/64/128스텝 비교까지 완료했다. 행동 추적에서 동결/해제 반복이 주요 낭비로 확인됐다. 상세 결과는 `docs/GROK_HANDOFF_003.md`에 있다.

추가 학습 없는 선택 방식 비교 240게임도 완료했다. 확률적 선택은 동결 반복을 줄였지만 성능 개선으로 이어지지 않았고, 정책 분포가 거의 균등한 상태임을 확인했다. `docs/SELECTION_RESULT_004.md`에 근거와 다음 실험을 기록했다.

512/2048/8192스텝 × 학습 seed 3개의 신규 학습 9회와 초기/최종 평가 1,440게임을 완료했다. 정책은 변했지만 학습량 증가만으로 안정적인 성능 개선은 확인되지 않았다. 수치·그래프·원본 검증과 과거 원격 실행의 폴더 오류 재시도 이력은 `docs/LEARNING_CURVE_RESULT_006.md`에 기록했다.

8192스텝에서 PPO epoch만1→4로 늘린 비교도 완료했다. 세 학습 seed의 확률적 선택 평균 보상은-5.90→-2.41, 생존율은0%→75%로 개선됐다. 새 평가 세트 검증 전의 후보 결과이며, 상세 비교와480게임 검증은 `docs/EPOCH_COMPARISON_RESULT_007.md`에 있다.

후속1040게임에서는 새로운 seed와 변경된 상대 구성 모두에서 개선이 유지됐다. epoch4 확률적 생존율은71.67%/73.33%였고 epoch1은 모두0%였다. 작은 환경의 연구 기준 설정은 `configs/research_train.json`, 결과·한계와 다음 학습 상대 다양화 단계는 `docs/HOLDOUT_RESULT_008.md`에 기록했다.

## 프로젝트 구조

실행 결과는 `runs/`에 별도로 보존한다.

```text
docs/          규칙 명세, 설계 결정, 검증 결과
src/           게임 엔진과 학습·평가 코드
tests/         단위·회귀·통합 테스트
data/          출처와 버전이 기록된 카드 데이터
configs/       학습·평가 설정
scripts/       환경 확인, 실행, 재개, 평가 명령
experiments/   실험별 명세
runs/          실행별 모델·로그·지표·메타데이터
```

`docs/GROK_BOT.md`와 `docs/GROK_HANDOFF_*.md`는 과거 원격 운영 기록이다. 현재 계획 실행은 로컬 WSL 경로를 우선한다.
