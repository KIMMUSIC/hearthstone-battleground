# Grok Run 003 실행 명세

Codex가 전달한 새 스냅샷만 사용한다. 기존 snapshot-001 및 venv311은 보존한다. 설치·삭제·소스 수정·보상 변경·추가 학습 금지. Windows 명령은 실행하지 않는다. 아래 명령은 Grok 클라우드 Linux에서만 실행한다.

1. ZIP SHA256과 SNAPSHOT_FILES.json의 개별 해시를 검증하고 비어 있는 `/workspace/hsai-rebuild/snapshot-003/` 아래에 압축을 푼다. 압축 내부 경로가 대상 밖으로 벗어나면 중단한다. `HearthStoneAI-Rebuild` checkout을 현재 작업 디렉터리로 사용한다.
2. 기존 Python `/workspace/hsai-rebuild/snapshot-001/venv311/bin/python`을 사용한다. 새 환경 설치는 하지 않는다. doctor, pytest 전체, Ruff를 실행하고 stdout/stderr/정확한 argv와 종료 코드를 보존한다. 게이트 전체 시간 180초. 실패 시 중단한다.
3. 추가로 외부 `timeout --kill-after=2s 60s` 아래에서 `tests/test_supervise.py -v`를 실행하고 잔여 테스트 프로세스가 없는지 확인한다. 감독기 시험 실패 시 학습하지 않는다.
4. 아래 JSON을 새 `campaign-003.json`에 기록한다. 감독기 전체 예산은 180초, 샘플링 RSS 2GiB다. CLI는 첫 실패 시 비정상 종료한다. 감독기를 임의로 수정하거나 이전 Bash wrapper로 대체하지 않는다.

```json
{
  "timeout_seconds": 180,
  "rss_limit_bytes": 2147483648,
  "commands": [
    ["/workspace/hsai-rebuild/snapshot-001/venv311/bin/python", "scripts/diagnostic_campaign.py", "--output", "runs/diagnostic-003"]
  ]
}
```

```sh
/workspace/hsai-rebuild/snapshot-001/venv311/bin/python scripts/supervise.py --spec campaign-003.json --output runs/supervisor-003
```

이는 seed 7의 64스텝 새 학습과 64스텝 추가 재개, 미학습·64·128스텝·heuristic·random의 seeds 201–220 평가만 수행한다. 실제 자원 한도 초과·오류가 발생하면 중단하고 실패 결과를 반환한다. RSS는 하드 상한이 아니며 샘플링 감시임을 보고서에 명시한다. 총 추가 디스크 100MiB를 넘기지 않는다. 기존 파일을 지워 예산을 맞추지 않는다.

5. 전후 SNAPSHOT 해시와 모든 checkpoint SHA256, 환경 및 freeze, 모든 게이트 원본 로그, `runs/supervisor-003/`, `runs/diagnostic-003/`를 반환한다. 평가 JSON의 trace도 포함한다. 결과 ZIP 이름은 `grok-diagnostic-003.zip`; 이 대화에 파일과 SHA256을 첨부한다. CopyFromBox나 Windows 로컬 경로 접근을 사용하지 않는다. 실패한 실행 기록도 보존한다.

성능 향상을 가정하지 않는다. 반환값을 Codex가 검증한 뒤에만 다음 실험을 정한다.
