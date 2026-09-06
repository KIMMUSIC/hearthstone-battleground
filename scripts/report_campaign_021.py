"""Publish 9/21-22 evidence only after the full campaign audit passes."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["training_steps"] != 300000:
        raise ValueError("The full campaign must pass before publication")
    relative = run.relative_to(root).as_posix()
    decision = audit["decision"]
    table = "| seed | 스텝 | 보상 | 전투 승률 | 생존율 | 동결/게임 | 강제 종료/게임 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for seed, rows in audit["reports"].items():
        for step, result in rows.items():
            metrics = result["mean_metrics"]
            table += (f"| {seed} | {step} | {result['mean_reward']:.3f} | "
                      f"{result['combat_win_rate']:.2%} | {result['survival_rate']:.2%} | "
                      f"{metrics['freezes']:.2f} | {metrics['forced_end_turns']:.2f} |\n")
    report21 = f"""# 2026-09-21 — 10만 스텝 규모 캠페인

상태: **완료**. 실제 수행일 2026-09-06. 실행 ID `{run.name}`.

9/20 사전 명세를 유지했다. seed7·17·27 각각 신규100,000스텝, 총300,000스텝을 완료했다. seed별8,192×12+1,696의13세그먼트, 총39세그먼트다. 각 seed 최종 모델 ZIP 카운터100,000/update12,500을 검증했다. 모든 재개 경계에서 직전 모델의 정책·옵티마이저와 다음 초기 checkpoint의 실제 텐서가 일치한다. 환경/RNG는 세그먼트에서 재시작하며 연속 궤적 동일성을 주장하지 않는다.

WSL Ubuntu 전용venv에서 고정 실행본의 `scripts/execute_campaign_021.py`를 한 번 실행했다. 39학습+9평가 감독 단계48개 모두 completed/exit0/잔여PID없음이다. 총 학습 wall {audit['elapsed_seconds']['train']:.1f}초, 개발 평가 wall {audit['elapsed_seconds']['evaluate']:.1f}초. 최대 표본RSS {audit['peak_sampled_rss_bytes']/1024**2:.1f}MiB, 출력과 증분ZIP {audit['output_and_return_bytes']/1024**2:.1f}MiB다. CPU1·개별학습60초·감독180초·표본RSS2GiB·전체출력ZIP1GiB·누적학습1200초/평가600초/전체1800초를 준수했다.

중복 실행과 실패 후 계속 실행을 차단하고 입력 무결성·실제 카운터·모델 변경·증분 반환ZIP을 감사했다. r1은 실행 전 검토용 패키지로 보존하고 수정 실행기는r2로 고정했다.

[감사 JSON](../../{relative}/audit.json), [실행 입력](../../handoff/{run.name}/execution.json), [진행 로그](../../{relative}/outputs/progress.json).

계획 유지: 초기0/중간49,152/최종100,000의 개발 평가900게임으로9/22 결론을 작성했다. 다음 분기는 사전 수치 기준으로 결정한다.
"""
    report22 = f"""# 2026-09-22 — 학습 곡선과 일반화 진단

상태: **완료**. 실제 수행일 2026-09-06. [9/21 실행](2026-09-21.md)의 동일 환경·동일 개발 세트로3개seed×3개시점×100게임=900게임을 평가했다.

{table}
세 seed 평균 최종−초기 보상은 **{decision['mean_reward_delta']:.4f}**, 양수 seed는 **{decision['positive_seeds']}/3**, 생존율 평균 차이는 **{decision['mean_survival_delta']:.4f}**, heuristic 보상3.4 대비 최종 평균 차이는 **{decision['mean_heuristic_reward_gap']:.4f}**다. 사전 학습 진행 기준 통과: **{decision['learning_progress']}**. 100만 스텝 후보 기준 통과: **{decision['million_step_candidate']}**.

독립 학습 반복은3개다. 보상차의 seed 재표집95% 기술 구간은 {decision['reward_delta_descriptive_bootstrap_95']}이며,3개 반복으로 넓은 모집단 신뢰를 주장하지 않는다. 900개 평가 에피소드를900개 독립 학습 반복으로 취급하지 않았다.

모델 해시는 평가 전후 불변이다. 모든 행동의 합법성·보상·승무패·리롤·동결·스왑·강제 종료 요약과 trace를 대조했다. 개발 상대 해시는020 기준선과 같고 최종 holdout은 번들에 없으며 읽거나 평가하지 않았다. [원본 감사/전체 지표](../../{relative}/audit.json)에 피해량·행동수도 보존했다.

![학습 곡선](../../{relative}/learning-curve.png)

9/23은 위 사전 기준을 그대로 적용해 증액 또는 제한 가설 검증을 결정한다. 이전019에서 발견한 행동별 할인·동결 지연 가설은 [통제 반례](../../experiments/diagnostics/discount-controlled-counterexample.json)에 보존했다. 이 반례가 학습 정체의 단일 원인이라는 주장은 후속 실험 전까지 하지 않는다.
"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = [("mean_reward", "Mean episode reward"), ("survival_rate", "Survival rate"),
               ("freezes", "Freeze actions / episode"), ("forced_end_turns", "Forced turns / episode")]
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        for seed, rows in audit["reports"].items():
            steps = sorted(map(int, rows))
            values = [rows[str(step)].get(key, rows[str(step)]["mean_metrics"].get(key)) for step in steps]
            axis.plot(steps, values, marker="o", label=f"seed {seed}")
        if key == "mean_reward":
            axis.axhline(3.4, color="gray", linestyle="--", label="heuristic baseline")
        axis.set_title(title)
        axis.set_xlabel("Training steps")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Fixed development evaluation: 100 games per checkpoint, 3 training seeds")
    fig.savefig(run / "learning-curve.png", dpi=160)
    plt.close(fig)
    (root / "docs/daily/2026-09-21.md").write_text(report21, encoding="utf-8")
    (root / "docs/daily/2026-09-22.md").write_text(report22, encoding="utf-8")
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
