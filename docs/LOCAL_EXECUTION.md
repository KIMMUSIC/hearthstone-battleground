# 로컬 WSL 실행 계약

2026-09-06 사용자 요청으로 Grok Bot 실행을 로컬로 전환했다. 기존 원격 결과는 역사적 증거로 보존하고 새 실행은 독립적인 ID를 사용한다.

- 개발: `D:\HearthStoneAI-Rebuild`. 원본 `D:\HearthStoneAI`는 읽기 전용.
- 실행: WSL Ubuntu, `/home/hwa3060/hsai-local/diversity-009-local-r1`.
- 가상환경: `/home/hwa3060/hsai-local/venv311/bin/python`, CPython 3.11.15.
- 의존성: `requirements-wsl.lock`은 실제 설치 버전. torch 2.10.0+cpu는 공식 CPU wheel 인덱스에서 설치했다. Windows 전역 환경은 변경하지 않았다.
- 고정 입력: `handoff/diversity-009-local-r1/FILES.json`. 이전 r1 ZIP을 기반으로 실행기·관련 테스트·execution.json만 로컬용으로 갱신했다. 게임 소스 SHA256은 `81b0dae34dfbac6bf875d1ebcf2647f34e68817b1b0987bc63685801e614cd90` 그대로다.
- CPU 1스레드, 감독기 단위 180초, 표본 RSS 2GiB, 출력+ZIP 1GiB, 개별 학습 협력적 60초. RSS는 하드 메모리 제한이 아니다. 기존 출력은 보존하고 자동 재시작하지 않는다.

## 실행

```powershell
wsl -d Ubuntu -- /home/hwa3060/hsai-local/venv311/bin/python -B /home/hwa3060/hsai-local/diversity-009-local-r1/scripts/run_diversity_009.py smoke
```

다음 단계는 `train`, 학습 감사 후 `reference`, `shifted` 순서다. 위 명령은 이미 완료된 단계에 재사용하지 않는다. 현재 상태는 일일 결과 문서를 확인한다. 실행기는 선언된 가상환경 진입 경로를 검사하며 심볼릭 링크를 실파일 호출로 바꾸지 않는다.

Windows 결과 보관 위치는 `runs/diversity-009-local-r1`이며 WSL 출력과 반환 ZIP을 복사한 뒤 해시·실제 카운터·평가를 감사한다. Linux 프로세스 감독기는 그대로 사용한다.

## 사전 검증

uv pip check 통과. WSL에서 107개 테스트 통과(15.80초). 두 조건 각각 64스텝 smoke와 모델 저장·로드·평가 완료. smoke 전체 23.63초, 테스트 최대 표본 RSS 799,354,880바이트, smoke 학습 프로세스 399,425,536바이트. 각 프로세스 그룹의 잔여 PID 없음.

새 환경이므로 과거 원격과 비트 단위 동일성을 주장하지 않는다. 비교 조건 두 개는 같은 로컬 환경·동일 seed 초기 파라미터로 검증한다.

## 완료 상태 — 2026-09-06

smoke → train → reference → shifted를 모두 완료했다. 총 49,152 학습 스텝과 2,000 평가 게임의 감사 결과는 `runs/diversity-009-local-r1/audit.json`의 passed / errors=[]이다. 각 감독기 종료 코드 0, 잔여 PID 없음이며 기존 실행을 반복하지 않는다. [실험 결과](OPPONENT_DIVERSITY_RESULT_009.md)에 따라 diverse 후보를 기각하고 fixed 기준선을 유지한다.

후속 개발 소스는 archival-subset-v4 / onehot-slots-v4를 사용한다. 신규 실행은 `/home/hwa3060/hsai-local/expansion-019-local-r1`, 고정 입력은 `handoff/expansion-019-local-r1/FILES.json`, 결과 보관은 `runs/expansion-019-local-r1`이다. `scripts/run_expansion_019.py`의 train → evaluate → baseline 순서로 감독기를 적용하며 CPU1·협력적 60초·감독기 180초·표본 RSS2GiB·출력 및 모든 반환 ZIP 1GiB 한도를 유지한다. 기존 64스텝 v3 진단은 새 실행의 완료 증거가 아니다. 각 단계는 한 번만 실행하고 실패해도 같은 ID로 재시작하지 않는다. 최종 상태는 [9/19](daily/2026-09-19.md), [9/20](daily/2026-09-20.md)를 따른다.

019 train → evaluate → baseline은 모두 완료했다. 8,192스텝·80게임·기준선 200게임·재현 6게임의 감사는 `runs/expansion-019-local-r1/audit.json` passed / errors=[]이다. 최대 표본 RSS397.3MiB, 출력과 모든 반환 ZIP61.0MiB, 종료 코드0·잔여 PID 없음. 019 실행기를 재실행하지 않는다. 다음 9/21 명세는 `experiments/EXPANSION_020_CAMPAIGN.md`에 있으며 누적 상한과 증분 ZIP을 갖춘 새 실행기가 필요하다.
