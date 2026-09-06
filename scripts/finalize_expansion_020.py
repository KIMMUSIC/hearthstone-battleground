"""Record measured 019/020 evidence and the unexecuted follow-up design."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def write(name, text):
    (ROOT / name).write_text(text, encoding="utf-8")


def main():
    prefix = "runs/expansion-019-local-r1"
    audit = read(prefix + "/audit.json")
    assert audit["status"] == "passed"
    train = read(prefix + "/outputs/train/summary.json")
    evaluation = read(prefix + "/outputs/evaluate/summary.json")
    spec = "experiments/EXPANSION_020_CAMPAIGN.md"
    draft = (ROOT / spec).read_text(encoding="utf-8")
    assert "## 실측 기반 확정" not in draft, "Already finalized; preserve first design"
    draft_hash = sha(spec)
    write("experiments/archive/expanded-011-diagnostic/expansion020-pre-baseline-design.md", draft)
    elapsed = audit["supervision"]["train"]["commands"][0]
    overhead = elapsed["ended_unix"] - elapsed["started_unix"] - train["train_seconds"]
    train_estimate = train["train_seconds"] * 300000 / 8192 + max(overhead, 0) * 39
    eval_estimate = max(v["seconds"] / 20 for v in evaluation["timings"].values()) * 900
    max_report_bytes_per_game = max(
        (ROOT / prefix / "outputs/evaluate" / f"{p}.json").stat().st_size / 20
        for p in evaluation["reports"]
    )
    train_bytes = sum(p.stat().st_size for p in (ROOT / prefix / "outputs/train").rglob("*") if p.is_file())
    # Twice the raw measurement covers an uncompressed return copy; input ZIP is stored once.
    disk_estimate = int(2 * (max_report_bytes_per_game * 900 + train_bytes * 39) +
                        (ROOT / "handoff/expansion-019-local-r1.zip").stat().st_size)
    inputs = ["configs/expansion019_train.json", "configs/expansion020_eval.json",
              "configs/expansion020_dev_opponents.json",
              "experiments/reserved/expansion020_final_eval.json",
              "experiments/reserved/expansion020_final_opponents.json"]
    planned = {
        "status": "designed-not-executed", "execution_scope": "9/21-22 separate goal",
        "compatibility": audit["compatibility"],
        "pre_baseline_design_sha256": draft_hash,
        "input_sha256": {name: sha(name) for name in inputs},
        "training_seeds": [7, 17, 27], "steps_per_seed": 100000, "total_steps": 300000,
        "segment_additional_steps": [8192] * 12 + [1696],
        "evaluation_milestones": [0, 49152, 100000], "expected_updates_per_seed": 12500,
        "base_training_config": "configs/expansion019_train.json",
        "training_opponent_mode": "fixed-v1", "development_games_per_model": 100,
        "development_model_games": 900, "final_games_per_frozen_seed_model": 100,
        "final_seed_range": [30001, 30100], "final_evaluation_executed": False,
        "acceptance": {"mean_seed_reward_delta_min": 0.5, "positive_seeds_min": 2,
                       "mean_seed_survival_delta_min": -0.05,
                       "million_step_candidate_heuristic_reward_gap_min": -0.5},
        "limits": {"cpu_threads": 1, "train_seconds_per_segment": 60,
                   "supervisor_seconds_per_phase": 180, "sampled_rss_bytes": 2 * 1024**3,
                   "all_outputs_and_return_zip_bytes": 1024**3,
                   "campaign_train_wall_seconds": 1200, "campaign_dev_eval_wall_seconds": 600,
                   "campaign_total_wall_seconds": 1800, "automatic_retry": False},
        "measured_estimates": {"training_wall_seconds": train_estimate,
                               "development_eval_seconds": eval_estimate,
                               "combined_seconds": train_estimate + eval_estimate,
                               "raw_outputs_plus_uncompressed_return_copy_bytes": disk_estimate,
                               "basis": prefix + "/audit.json",
                               "uncertainty": "One 8192-step run; not a completion guarantee"},
        "packaging": "One frozen input bundle; incremental per-segment return ZIPs, no cumulative duplication",
        "resume_contract": "Weights/optimizer/counters restored; episodes and RNG restart at each boundary",
        "stop_conditions": ["source/compatibility mismatch", "non-finite parameters",
                            "missing or partial expected steps", "supervisor or cumulative budget",
                            "disk limit", "evaluation trace mismatch"],
    }
    assert sum(planned["segment_additional_steps"]) == 100000
    assert all(n % 32 == 0 for n in planned["segment_additional_steps"])
    assert disk_estimate < 1024**3
    write("experiments/expansion020_campaign.json", json.dumps(planned, indent=2) + "\n")
    appendix = f"""
## 실측 기반 확정 — 2026-09-06

019는 8,192스텝/1,024 update를 달성했다. 학습 함수 {train['train_seconds']:.3f}초, 약 {train['training_steps_per_second']:.1f}스텝/초다. 초기 모델 로드 비용까지 포함한 느린 평가 측정은 게임당 {max(v['seconds']/20 for v in evaluation['timings'].values()):.4f}초다. 동일 환경의 단순 외삽은 학습·프로세스 시작 약 {train_estimate:.1f}초, 개발 모델 평가 약 {eval_estimate:.1f}초, 합계 약 {(train_estimate+eval_estimate)/60:.1f}분이다. 한 번의 짧은 실행이므로 완료 시간을 보장하지 않는다.

캠페인 누적 중단 상한은 학습 wall 1,200초, 개발 평가 wall 600초, 전체 wall 1,800초(30분)로 명세한다. 개별 60초/180초 한도와 함께 적용하며 더 먼저 닿는 한도로 중단한다. 39개 세그먼트를 모두 180초씩 허용하는 별도 증액이 아니다. 실패·부분 학습·시간 도달 시 결과를 보존하고 다음 세그먼트를 시작하지 않는다. 이 계획만으로 다음 날짜 실행을 시작하지 않는다.

디스크는 019 최대 모델 평가 파일/게임과 학습 출력/세그먼트를 적용하고 반환 사본까지 두 배로 잡아 약 {disk_estimate / 1024**2:.1f}MiB다. 입력 번들은 한 번 보관하고 세그먼트별 반환 ZIP은 신규 출력만 포함한다. 019의 누적 ZIP 방식을 39번 반복하지 않는다. 9/21 실행기는 이 증분 포장과 누적 시간·디스크 중단을 먼저 검증해야 한다.

입력·규칙·소스 해시와 정확한 세그먼트, 평가 seed, 수치 기준은 [기계 판독 명세](expansion020_campaign.json)에 고정했다. 최종 평가 입력은 실행 번들 밖 `experiments/reserved`에 보존했다. 스키마·ID·해시만 점검했으며 성능 결과를 생성하지 않았다.

019 최종 모델은 20게임 모두 동결 행동만 반복했다(게임당 144행동·강제 종료 6회). 초기·최종 보상은 모두 -6.0이다. 100k는 학습량에 따른 이 실패 행동의 변화 여부를 보는 제한 실험이며 성능 개선을 확인한 증액이 아니다. 기존 수치 판정은 유지하고, 9/22에서 동결 비율·강제 종료·보상 곡선을 확인한 뒤 개선이 없으면 9/23 원인 진단을 우선한다. 100만 스텝은 자동 실행하지 않는다.
"""
    write(spec, draft.replace("상태: 준비 중. 최종 소스·분리 평가 입력 해시와 실측 예산은 9/19 완료 후 확정한다.",
                             "상태: 명세 완료. 최종 소스·분리 평가 입력 해시와 실측 예산을 확정했다.") + appendix)
    max_rss = max(c["peak_sampled_rss"] for v in audit["supervision"].values() for c in v["commands"])
    report19 = f"""# 2026-09-19 — 확장 환경 통합·실측 및 2주 보고

상태: **완료**. 실제 수행일 2026-09-06. 실행 ID `expansion-019-local-r1`.

## 2주간 지원 범위

009의 fixed/diverse 비교와 재현 감사를 마치고 fixed 기준선을 유지했다. 일반 허수아비골렘의 사망·소환, 토큰 풀 제외, 관측·매핑 버전 검사를 구현했다. 9/17에서 토큰·중복 후보를 제외한 상점 7종·소환 토큰 1종을 확정했고, 9/18 경제 경계를 검증했다. 현재 계약은 archival-subset-v4 / onehot-slots-v4, 행동 37개다. 전체 pytest **156 passed in 18.74s**, ruff `src scripts tests` 통과. 동결 재현 지표 보완 후 관련 6개 테스트도 통과했다.

전체 하스스톤 규칙·8인 경기·실전 강도를 구현한 단계는 아니다. 골렘 황금·트리플, 뱃사람 즉시공격은 미지원이다. 단순 카드 황금은 문서화된 연구용 2배 규칙을 따른다.

## 실제 로컬 실행

WSL Ubuntu `/home/hwa3060/hsai-local/venv311/bin/python`에서 고정 실행본의 `scripts/run_expansion_019.py train`, 다음 `evaluate`를 각 한 번 실행했다. 두 단계 모두 종료 코드 0, 감독기 completed, 잔여 PID 없음이다. CPU1·협력적 학습 60초·감독기 단계 180초·표본 RSS2GiB·출력/전체 ZIP1GiB 한도를 적용했다.

- 신규 학습 **8,192스텝**, 실제 update **1,024**, 초기 카운터 0. 학습 함수 **{train['train_seconds']:.3f}초**, 처리량 **{train['training_steps_per_second']:.1f}스텝/초**. 시간 중단 없이 목표 스텝으로 종료했다.
- 초기·최종 checkpoint를 저장·로드하고 실제 ZIP 카운터와 메타데이터를 검사했다. 12개 파라미터 텐서가 바뀌었고 모두 유한하다.
- 초기/최종 모델, heuristic, random 각 20게임, 합계 **80게임**. seed 19001–19020과 별도 상대 입력을 사용했다.
- 020 기준선을 포함한 세 단계 최대 표본 RSS **{max_rss / 1024**2:.1f}MiB**. 출력과 모든 반환 ZIP 합계 **{audit['output_and_all_zip_bytes'] / 1024**2:.1f}MiB**. 표본 RSS는 하드 제한을 뜻하지 않는다.

| 정책 | 평균 보상 | 전투 승률 | 생존율 |
| --- | ---: | ---: | ---: |
| 초기 모델 | -6.00 | 0% | 0% |
| 8,192스텝 모델 | -6.00 | 0% | 0% |
| heuristic | 5.15 | 71.25% | 100% |
| random | -5.65 | 2.50% | 0% |

최종 모델은 게임당 144회 동결을 반복하고 6번 강제 턴 종료됐다. 학습·저장·평가 경로는 통과했지만 성능 개선은 없으며, 짧은 단일 seed 결과로 일반화 성능을 주장하지 않는다.

## 무결성과 후속 계획

[감사 JSON](../../{prefix}/audit.json)은 passed / errors=[]이다. 입력 68개 파일과 반환 ZIP 3개 CRC·해시·내부 입력을 검사했다. 모든 평가 행동은 합법이고 trace와 보상·전투 결과·행동 수·동결·리롤·스왑·강제 종료 요약이 일치한다. 최종 모델 SHA256 `{train['final_model_sha256']}`, 소스 SHA256 `{audit['compatibility']['source_sha256']}`. 전체 서명은 감사 파일에 있다.

실행 전 입력은 [FILES](../../handoff/expansion-019-local-r1/FILES.json), 실행 명령은 [execution.json](../../handoff/expansion-019-local-r1/execution.json), 원시 출력은 [train summary](../../{prefix}/outputs/train/summary.json), [평가 summary](../../{prefix}/outputs/evaluate/summary.json)에 보존한다. ZIP 내부 당일 control/result에는 아직 생성하지 않은 자기 ZIP 해시가 없고, 디스크의 최종 result에 추가되는 구조를 감사에서 명시적으로 검증했다.

계획 변경: v3의 64스텝 진단을 완료 근거에서 제외하고 v4 감독 실행으로 보완했다. 9/20은 [실측 기반 캠페인 명세](../../experiments/EXPANSION_020_CAMPAIGN.md)를 채택한다. 9/21 학습은 이번 범위에서 시작하지 않는다.
"""
    write("docs/daily/2026-09-19.md", report19)
    b = audit["reports"]
    table = "| 지표(게임당 평균) | heuristic | random |\n| --- | ---: | ---: |\n"
    for label, key in (("승리", "wins"), ("무승부", "draws"), ("패배", "losses"),
                       ("받은 피해", "damage_taken"), ("준 피해", "damage_dealt"),
                       ("행동 수", "actions"), ("강제 턴 종료", "forced_end_turns"),
                       ("리롤", "rerolls"), ("스왑", "swaps"), ("동결", "freezes")):
        table += f"| {label} | {b['baseline/heuristic']['mean_metrics'][key]:.2f} | {b['baseline/random']['mean_metrics'][key]:.2f} |\n"
    write("docs/daily/2026-09-20.md", f"""# 2026-09-20 — 개발 기준선과 다음 캠페인 설계

상태: **완료**. 실제 수행일 2026-09-06. 9/17–20 연속 범위를 완료했고 9/21은 미착수다.

## 기준선

v4 고정 실행본의 `scripts/run_expansion_019.py baseline`을 로컬 WSL 감독기에서 실행했다. 종료 코드 0, completed, 잔여 PID 없음. 개발 seed 20001–20100으로 heuristic/random 각 **100게임**, 별도 3개 seed씩 **6게임 재현 검사**를 마쳤다. 재현 검사에서는 전체 에피소드 지표와 행동 trace가 정확히 일치했다. 두 정책의 모든 행동은 합법이며 종료했다. 기존 규칙 기반 정책은 구매·플레이·교체·업그레이드·트리플 후보·발견을 처리한다. 동결·스왑을 사용하지 않는 현재 우선순위를 기준선으로 고정하며 최적 전략을 주장하지 않는다.

| 정책 | 게임 수 | 평균 보상 | 전투 승률 | 생존율 |
| --- | ---: | ---: | ---: | ---: |
| heuristic | 100 | 3.40 | 58.50% | 100% |
| random | 100 | -5.90 | 0.1664% | 0% |

{table}
평균 보상은 전투·생존·피해 지표와 별도로 해석한다. 행동 사용량이 모두 낭비를 뜻하지는 않으며 강제 턴 종료와 019 모델의 동결 반복을 실패 행동 근거로 본다.

## 분리·예산·판정

학습은 fixed-v1, 019 실행 검사는 seed 19001–19020, 개발 평가는 20001–20100, 최종 평가는 별도 30001–30100이다. 최종 상대·seed는 `experiments/reserved`에 보관하고 019 실행 번들에서 제외했다. 스키마·카드 ID·해시만 확인했고 최종 성능 평가는 하지 않았다. 개발/최종 입력을 학습에 읽지 않는 회귀 검사를 통과했다.

다음 실험은 seed **7·17·27**, 각각 신규 **100,000스텝**이다. 8,192×12+1,696 세그먼트로 동일 경계에서 재개하며 초기/49,152/100,000 지점을 개발 세트 각 100게임으로 비교한다. 학습량 이외의 보상·카드·상대 분포는 바꾸지 않는다. 실제 예상 시간 약 **{(train_estimate+eval_estimate)/60:.1f}분**, 누적 상한 **30분**(학습 20분/개발 평가 10분)이며 개별 60초·감독기 180초·CPU1·표본 RSS2GiB·총 출력/ZIP1GiB도 유지한다. 중단 시 부분 결과를 보존한다.

수치 기준은 결과 전에 작성한 명세를 유지했다: 세 seed 평균 최종−초기 보상 ≥+0.5, 최소 2개 seed 양수, 생존율 평균 차이 ≥−0.05. 증액 후보는 추가로 heuristic 대비 보상 차이 ≥−0.5가 필요하다. 통과해도 100만 스텝 자동 실행은 아니다. 019 모델은 개선이 없었으므로 100k 결과의 동결 비율·강제 종료를 확인하고 개선이 없으면 9/23 원인 진단을 우선한다.

## 증거와 다음 재개

- [감사](../../{prefix}/audit.json), [기준선 원시 요약](../../{prefix}/outputs/baseline/summary.json)
- [캠페인 명세](../../experiments/EXPANSION_020_CAMPAIGN.md), [정확한 입력 해시·예산 JSON](../../experiments/expansion020_campaign.json)
- [9/17 카드 교정](2026-09-17.md), [9/18 경제 검증](2026-09-18.md), [9/19 2주 보고](2026-09-19.md)

계획 변경: 토큰·중복을 포함한 v3 기준선과 10,800초 설정은 진단 이력으로 보존하고 사용하지 않는다. 9/21은 새 캠페인 실행기에 누적 예산·증분 ZIP·세그먼트 재개 검증을 추가한 뒤 고유 실행본으로 시작한다. 019 완료 단계를 재실행하거나 기존 모델에 덮어쓰지 않는다.
""")
    write("docs/daily/2026-09-17.md", (ROOT / "docs/daily/2026-09-17.md").read_text(encoding="utf-8").replace(
        "상태: 최종 교정 검증 중.", "상태: **완료**. 최종 v4 전체 156개 테스트와 ruff 검증 통과."))
    print(json.dumps(planned["measured_estimates"]))


if __name__ == "__main__":
    main()
