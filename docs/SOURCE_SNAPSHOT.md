# 첫 Git 소스 스냅샷 — 2026-09-06

사용자가 지정한 `KIMMUSIC/hearthstone-battleground` 저장소에 현재 Rebuild 코드를 보존하기 위한 첫 소스 반입이다. 초기 확인 시 원격에 ref/commit이 없었고 비공개였다. 기존 프로젝트 `D:/HearthStoneAI`는 수정하지 않았다.

## 포함 범위

- `src/`, `scripts/`, `tests/`: 현재 엔진·학습·평가·실험 도구와 검증 코드.
- `configs/`, `data/`: 제한된 카드 데이터, 매핑, 실행 설정.
- `experiments/`, `docs/`: 사전 명세, 진단 결과, 규칙과 개발 기록.
- `docs/blog/`: 미게시 시각 중심 초안과 그 수치의 작은 집계 근거.
- 프로젝트 메타데이터, 의존성 lock, README와 개발 지침.

`runs/`, `handoff/` 전체, 모델·압축 전달물, 캐시, 로컬 환경·에이전트 상태, 원본 대용량 데이터는 Git에서 제외한다. 로컬 파일은 삭제하지 않는다. 과거 결과 문서에 있는 `runs/`·`handoff/` 링크는 로컬 아카이브 위치를 뜻하며 checkout에는 없다. 과거 모델을 재현하려면 해당 기록의 해시에 맞는 아카이브가 별도로 필요하다. 현재 모델 파일의 소스 호환성 검사는 우회하지 않는다.

`.gitattributes`는 `-text`로 원본 바이트를 보존한다. 기존 파일에 LF와 CRLF가 혼재해 자동 정규화하면 소스·데이터·증거 파일의 해시가 달라지기 때문이다. 모든 반입 파일의 Git index와 로컬 원본 바이트를 대조한다. 기존 로컬 실행 사본이나 모델 메타데이터는 변경하지 않는다. 과거 데스크톱 UI 덤프(`docs/daily/*-ui.txt`)는 제외하고 결과 요약은 보존한다.

## Fresh 소스 검증

지원 실행 경로인 WSL Linux 전용 Python으로 수행했다.

```text
WSL /home/hwa3060/hsai-local/venv311/bin/python -B -m pytest -q
383 passed in 67.85s (exit 0)

python -m ruff check src tests scripts
All checks passed (exit 0)

python -B scripts/bgai.py doctor
python -B scripts/bgai.py smoke --seed 42
exit 0
```

Windows 전역 Python의 전체 테스트는 370 passed / 7 skipped / 6 failed였다. 6건은 Linux venv를 요구하는 실험 통합 테스트의 환경 검사에서 실패하며 WSL에서는 모두 통과했다. 이 6건의 Windows skip 처리는 아직 보완하지 않았다. 소스 동작 변경 없이 현재 지원 환경의 검증 결과와 제한을 함께 보존한다.

소스·문서·설정·실험 파일에서 일반적인 credential 및 private-key 패턴을 확인했다. 새로운 의존성 설치나 장시간 재학습은 수행하지 않았다. 이 검증은 시뮬레이터의 현재 지원 범위에 대한 것이며 실제 하스스톤 전체 규칙의 정확성이나 실전 AI 성능을 입증하지 않는다.

초안 검증: 그래프 집계 재계산 일치, HTML의 링크 21개 중 모든 로컬 파일·anchor 존재, 브라우저 조건 전환 71.67%↔73.33% 확인, 390px 모바일 viewport에서 가로 overflow 없음. 데스크톱·모바일 화면을 확인했으며 브라우저 오류 로그는 없었다. 이 HTML은 블로그 배포를 하지 않는 로컬 미리보기다.

첫 반입의 기존 문서·생성 SVG·Python 파일 하나에는 EOF 공백 등 `git diff --check` 경고가 있다. 역사적 산출물이나 코드 해시를 바꾸는 공백 정리는 이 작업에 포함하지 않는다. 이번에 만든 초안·Git 설정·스냅샷 문서는 별도로 whitespace 검사를 수행한다.
