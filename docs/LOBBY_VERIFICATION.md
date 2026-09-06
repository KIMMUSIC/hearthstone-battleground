# 8인 로비 지원 규칙과 검증 — 10/1 완료

실제 검증일: 2026-09-06. 규격은 [LOBBY_CONTRACT.md](LOBBY_CONTRACT.md), 실행 증거는 [030 감사](../runs/lobby-030-local-r1/audit.json)다.

| 지원 항목 | 회귀 근거 | 실제 실행 근거 |
| --- | --- | --- |
| 8좌석 모집·37행동·사망 후 행동 차단 | test_lobby.py, test_lobby_integration.py | 100경기 76,844행동 합법성·재현 |
| 동시 피해·탈락·공동 순위·순위 합36 | test_lobby.py, test_lobby_cross_cases.py | 100경기 자연 종료, 567탈락 이벤트 |
| 짝수 매칭·홀수 유령·최근 상대 회피 | test_lobby_matching.py, test_lobby_cross_cases.py | 2,214매칭·6,077전투 |
| 유한 풀·예약·교체·동결·구매·판매·탈락 반환 | test_lobby_pool.py, test_lobby.py | 매 행동 재고 범위·보존 검사 |
| 트리플3장·발견 예약·미선택 반환·고갈 | test_lobby.py, test_lobby_pool.py | 2,587트리플·2,516발견·63발견 고갈 |
| 골렘 전투 토큰·원본 보드와 풀 분리 | test_lobby_cross_cases.py | 제어된 실제 전투 검증 |
| 자기/공개 관측·공간·마스크 | test_lobby_observation.py, test_lobby_integration.py | 모든 행동 전 공간 검사 |
| 전역 ID·상대 비공개 상태 차단 | test_lobby.py, test_lobby_integration.py | 정책 view의 전역 ID 부재 감사 |
| 세션별 RNG·재현성 | test_lobby_matching.py, test_lobby_integration.py | 6정책 재실행·100전체 행동 trace 재현 |
| 자연 종료와 제한 중단 구분 | test_lobby.py, test_lobby_integration.py | 본 실행100/100 자연 종료; 중단은 별도 테스트 |
| 전투 예외 시 매칭 이력·RNG 복구 | test_lobby.py | 예외 주입 회귀; 마지막 경제 행동 유지, end만 허용 |

전체 회귀 **276 passed (15.66초)**, `ruff check src scripts tests` 통과. 전역 entity ID 누수와 전투 실패 전 매칭 이력 변경을 수정하고 회귀로 고정했다. 모든 비밀 채널을 수학적으로 증명했다는 의미는 아니다.

입력 ZIP SHA256: `79f8fc3745e2beaa67a2163a7e3a9fdc2133fe2787e8a9ac580eae525351cad2`.
반환 ZIP SHA256: `141a5a6e732a93fe229cc7b8f2d893e2f229dd7236a63b0e74f8f0cb9944079a` (CRC와105파일 해시 확인).
50seed × heuristic/mixed 두 구성의 100본 경기다. 추가6정책 재현과100행동 기록 재현은 독립 표본으로 집계하지 않는다. 감독18.49초, 샘플링 최고 RSS54,128,640바이트, 출력+반환33,591,302바이트로 한도 이내다. 고정 입력은 실행·감사 전후 동일하다.

상점7종·전투 토큰1종·최대2티어의 연구 규격이다. 카드별 풀15장은 이 부분 규격의 선택이다. 영웅·방어도·주문·현행 피해 상한·전체 카드·골렘 황금/트리플은 미지원이다. 최신 패치 일치를 주장하지 않으며 규칙 출처와 차이는 계약에 보존했다.

8인 PPO 어댑터는 10/2의 `lobby-032-local-r1` smoke에서 관측·보상·모델 호환성·짧은 학습·개발 평가 audit를 통과했다. 최종 모델은 개발 평가에서 heuristic 기준선보다 낮아 성능 개선 증거는 아니다. 완전 자기 대전 학습은 아직 미구현이다. 021/024 모델은 당시 고정 환경의 결과이며 8인 정책 성능의 증거가 아니다.

