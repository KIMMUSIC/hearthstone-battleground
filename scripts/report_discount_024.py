"""Write 9/24–26 results only from a passed frozen-run audit."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    result = json.loads((run / "audit.json").read_text())
    if result["status"] != "passed" or result["training_steps"] != 49152 or result["evaluation_games"] != 1200:
        raise ValueError("Complete audited campaign required")
    decision = result["decision"]
    rows = ["| 학습 seed | 조건 | 초기 보상 | 최종 보상 | 초기 생존율 | 최종 생존율 |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for seed, arms in result["reports"].items():
        for arm, reports in arms.items():
            initial, final = reports["initial"], reports["final"]
            rows.append(f"| {seed} | {arm} | {initial['mean_reward']:.3f} | {final['mean_reward']:.3f} | "
                        f"{initial['survival_rate']:.1%} | {final['survival_rate']:.1%} |")
    relative = run.relative_to(root).as_posix()
    daily24 = root / "docs/daily/2026-09-24.md"
    text = daily24.read_text(encoding="utf-8")
    text = text.replace("**진행 중 — 실제 감독 실행**", "**완료 — 실행 감사 통과**")
    text += (f"\n최종 결과: 실제49,152스텝·1,200게임,18감독 단계 완료. "
             f"학습 {result['driver']['phase_totals']['train']:.2f}초, "
             f"평가 {result['driver']['phase_totals']['evaluate']:.2f}초, "
             f"전체 {result['driver']['elapsed_seconds']:.2f}초. "
             f"최대 표본RSS {result['peak_sampled_rss']/2**20:.1f}MiB, "
             f"출력/반환 {result['output_and_return_bytes']/2**20:.1f}MiB. "
             f"[실행 감사](../../{relative}/audit.json)에서 실제카운터·gamma·유한가중치·입력해시·모델불변·"
             "paired 초기 정책 동일성·기준선 엔진 동일성·ZIP 전체 무결성을 확인했다. 재시도 없이 한 번 실행했다.\n")
    daily24.write_text(text, encoding="utf-8")
    report25 = ("# 2026-09-25 — 할인율 결과와 후보 동결\n\n상태: **완료**. 실제 수행일 2026-09-06.\n\n"
                + "\n".join(rows) + "\n\n"
                f"gamma1−gamma0.99 paired 최종 보상차 평균 **{decision['mean_paired_reward_delta']:.4f}**, "
                f"양수 seed **{decision['positive_paired_seeds']}/3**, "
                f"생존율 차이 **{decision['mean_paired_survival_delta']:.4f}**. "
                f"가설 지지 기준 통과: **{decision['hypothesis_supported']}**. "
                f"candidate의 heuristic 대비 보상차 **{decision['mean_heuristic_reward_gap']:.4f}**. "
                f"모든 후보 기준 통과: **{decision['candidate_selected']}**.\n\n"
                f"독립 반복은 세 학습 seed다. paired seed 재표집95% 기술 구간은 "
                f"{decision['paired_reward_delta_descriptive_95']}다. 1,200게임을 독립 학습 반복으로 취급하지 않는다. "
                "단일 가설·작은 학습량의 결과로 일반적인 gamma 우열을 주장하지 않는다.\n\n"
                f"[전체 감사와 수치](../../{relative}/audit.json). 평가 이전 고정한 "
                "[판정 기준](../../experiments/DISCOUNT_024.md)을 그대로 적용했다. "
                "기준선은 동일 개발 seed·상대의 heuristic 3.4이며, 실행본에 보존한 이전 FILES와 비교해 "
                "training.py 외 엔진·데이터·규칙이 같음을 검사했다.\n")
    if decision["candidate_selected"]:
        report25 += "\n후보 세 seed의 모델 해시를 별도 동결한 뒤9/26 최종 세트를 평가해야 한다. 아직9/26은 완료가 아니다.\n"
    else:
        report25 += "\n후보 없음으로 동결했다. control이나 이전 캠페인의 최고 seed를 대신 선택하지 않는다. 최종 평가를 실행하지 않는다.\n"
        report26 = ("# 2026-09-26 — 3주 결과와 최종 평가 보류\n\n상태: **완료 — 후보 없음 분기**. 실제 수행일2026-09-06.\n\n"
                    "021은 세 seed 각10만스텝에서 초기보다 개선됐지만 규칙 기반 기준선과의 차이로 증액 기준을 통과하지 못했다. "
                    "024의 단일 할인율 비교도 최종 후보 기준을 충족하지 못했다. "
                    "[9/25 결과](2026-09-25.md)에 모든 seed와 사전 판정을 남겼다. 임의의 승자를 만들지 않았다.\n\n"
                    "최종 세트는 두 실행 패키지에서 제외했고 최종 평가를 수행하지 않았다. 아직 미사용 최종 조건이며 "
                    "개발용으로 재분류하지 않는다. 모델 성능은 개발 조건에서만 측정됐으며 미사용 상대·실제 게임으로의 일반화는 입증되지 않았다.\n\n"
                    "현재 엔진의 지원 범위는 명시한7개 상점 카드와1개 소환 토큰,2티어,고정 상대의 제한 게임이다. "
                    "전체207개 회귀와 후속 실행기8개 테스트가 통과했고,024 감사에서 gamma 학습 설정 외 엔진·카드·규칙은 "
                    "019 검증본과 동일함을 확인했다. 정책의 약함은 엔진 검증 실패를 뜻하지 않는다. "
                    "따라서9/27–10/1의 별도8인 경기·공유 풀·누수 검증을 진행한다. "
                    "실제8인 경기 기능이 구현됐다는 주장은 아직 하지 않는다.\n")
        (root / "docs/daily/2026-09-26.md").write_text(report26, encoding="utf-8")
    (root / "docs/daily/2026-09-25.md").write_text(report25, encoding="utf-8")
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
