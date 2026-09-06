# Linux 실행 감독기

`scripts/supervise.py`는 Python 표준 라이브러리만 사용한다. JSON의 `commands`를 shell 없이 argv로 실행하고 첫 실패 시 중단한다. `timeout_seconds`는 개별 명령이 아니라 캠페인 전체 제한이다. 명령별 stdout/stderr, argv, PID, 시각, 종료 코드와 샘플링 최대 RSS를 새 출력 폴더에 보존한다. 기존 출력 폴더는 거부한다.

```json
{
  "timeout_seconds": 180,
  "rss_limit_bytes": 2147483648,
  "commands": [
    ["/path/to/venv/bin/python", "scripts/bgai_threaded.py", "doctor"]
  ]
}
```

프로젝트 checkout에서 실행한다. 상대 경로는 현재 작업 디렉터리 기준이다.

```sh
python scripts/supervise.py --spec campaign.json --output runs/supervisor-unique
```

실제 RL 프로세스는 `scripts/bgai_threaded.py`로 실행한다. 이 프로세스 안에서 Torch intraop/interop을 각각 1로 설정하고 로그 첫 줄에 PID와 실제 값을 출력한다. `train` 설정의 `threads`도 1이어야 한다. 기본 bgai 실행만으로 같은 계약이 성립한다고 가정하지 않는다.

감독기는 새 프로세스 그룹을 만들고 그룹 전체에 TERM, 이후 KILL을 보낸다. 살아 있는 그룹 구성원이 남거나 부모가 자식을 남기고 종료하면 실패다. 자식·손자를 `/proc`의 그룹 ID로 합산한다. 그룹/세션을 이탈하는 프로그램, 감독기에 대한 SIGKILL, 호스트 종료를 견디는 서비스는 지원하지 않는다. 좀비는 실행·메모리 사용이 없으므로 살아 있는 프로세스에서 제외하며, 직접 자식은 wait로 회수한다.

RSS는 20ms 간격의 관측값이며 엄격한 메모리 상한이 아니다. 순간 급증이나 샘플 사이에 종료된 프로세스의 최대 사용량을 보장하지 않는다. cgroup 하드 상한은 구현하지 않았다. 추가 패키지 설치 및 디스크 용량 제한도 이 감독기의 기능이 아니다.

실제 Linux 회귀 시험:

```sh
timeout --kill-after=2s 60s python3 tests/test_supervise.py -v
```

2026-09-05 Ubuntu WSL Python 3.12.3에서 6개 통과(1.691초). 정상 종료, exit 7 후 후속 작업 차단, TERM 무시 자식·손자 timeout 종료, 손자의 RSS 초과, 부모만 종료한 경우, 감독기 취소를 검증했다. 이 표준 라이브러리 운영 시험과 Python 3.11 RL 환경 검증은 별개다. 현재 로컬 WSL Python3.11.15 환경에서도 같은 감독기 시험을 포함한 107테스트를 통과했다. 현재 실행 계약은 LOCAL_EXECUTION.md를 따른다.
