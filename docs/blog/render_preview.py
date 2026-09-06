"""Render the review-only Markdown and export its figures; never publish."""

from pathlib import Path
import json
import re

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "font.size": 15,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#d6d9dc",
    "text.color": "#33383d",
    "axes.labelcolor": "#50565c",
    "xtick.color": "#50565c",
    "ytick.color": "#50565c",
})
BLUE, ORANGE, GRAY = "#477da8", "#b5643a", "#b2b8bc"


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")
ax.barh([1, 0], [256, 1024], color=[GRAY, BLUE], height=.46)
ax.set_yticks([1, 0], ["epoch 1", "epoch 4"])
ax.set_xlim(0, 1250)
ax.set_xticks([0, 256, 512, 768, 1024])
ax.set_xlabel("optimizer 업데이트 수", labelpad=12)
ax.tick_params(axis="y", length=0)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#eeeeee")
for y, value in [(1, 256), (0, 1024)]:
    ax.text(value + 25, y, f"{value:,}", va="center", fontweight="bold")
save(fig, "updates")

holdout = json.loads((ROOT / "evidence/holdout-008-audit.json").read_text())
survival = []
for condition in ("fresh-seeds", "shifted"):
    cells = [v["survival_rate"] for k, v in holdout["summary"].items()
             if k.startswith(condition + "-epoch-4-") and "-sample-" in k]
    assert len(cells) == 9
    survival.append(sum(cells) / len(cells) * 100)
fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
for offset, values, color, label in [(.24, [0, 0], GRAY, "epoch 1"),
                                     (0, survival, BLUE, "epoch 4"),
                                     (-.24, [100, 100], "#737e86", "휴리스틱")]:
    ys = [1 + offset, offset]
    ax.barh(ys, values, height=.2, color=color, label=label)
    for y, value in zip(ys, values, strict=True):
        ax.text(value + 2, y, f"{value:.2f}%" if value not in (0, 100) else f"{value:.0f}%",
                va="center", fontsize=14)
ax.set_yticks([1, 0], ["새 seed", "새 seed\n+ 상대 변경"])
ax.set_xlim(0, 125)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("에피소드 생존율 (%)")
ax.tick_params(axis="y", length=0)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#eeeeee")
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=3, frameon=False, fontsize=14)
save(fig, "survival")

# Experiment 044 was performed on 2026-09-06; daily filename is a plan label.
diagnosis = json.loads(
    (ROOT.parent.parent / "experiments/diagnostics/lobby044_visited_states.json").read_text()
)
dev_accuracy = [diagnosis["teacher_dev_model_metrics"][seed]["candidate"]["accuracy"] * 100
                for seed in ("7", "17", "27")]
visited_accuracy = [
    diagnosis["final_trace_replay"][seed]["candidate"]["visited_state_teacher_agreement"]
    ["accuracy"] * 100 for seed in ("7", "17", "27")
]
fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
for offset, values, color, label in [(.17, dev_accuracy, BLUE, "교사 dev 상태"),
                                     (-.17, visited_accuracy, ORANGE, "모델 방문 상태")]:
    ys = [2 + offset, 1 + offset, offset]
    ax.barh(ys, values, height=.28, color=color, label=label)
    for y, value in zip(ys, values, strict=True):
        ax.text(value + 2, y, f"{value:.2f}", va="center", fontsize=14)
ax.set_yticks([2, 1, 0], ["seed 7", "seed 17", "seed 27"])
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("휴리스틱 행동과의 일치율 (%)")
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eeeeee")
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False, fontsize=14)
save(fig, "state-distribution")

lobby = json.loads((ROOT / "evidence/lobby-045-audit.json").read_text())
fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
for y, seed in zip([2, 1, 0], ["7", "17", "27"], strict=True):
    before = lobby["games"][seed]["control"]["final"]["mean_rank"]
    after = lobby["games"][seed]["candidate"]["final"]["mean_rank"]
    ax.plot([before, after], [y, y], color="#b9bdc0", linewidth=2)
    ax.scatter(before, y, color=BLUE, s=55, label="추가 전" if y == 2 else None, zorder=3)
    ax.scatter(after, y, color=ORANGE, s=55, label="추가 후" if y == 2 else None, zorder=3)
    ax.text(min(before, after) - .05, y + .2, f"{before:.2f} → {after:.2f}",
            ha="right", fontsize=14)
ax.set_yticks([2, 1, 0], ["seed 7", "seed 17", "seed 27"])
ax.set_ylim(-.5, 2.6)
ax.set_xlim(.8, 8.3)
ax.set_xticks(range(1, 9))
ax.set_xlabel("평균 순위 (낮을수록 좋음)")
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eeeeee")
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=14)
save(fig, "rank")

raw = (ROOT / "draft.md").read_text(encoding="utf-8")
body = raw.split("---", 2)[2]
article = markdown.markdown(body, extensions=["tables", "fenced_code"])
article = re.sub(
    r'(<img alt="([^"]*)" src="([^"]*)" />)',
    r'<a href="\3" target="_blank" rel="noopener" aria-label="\2 크게 보기">\1</a>',
    article,
)
style = """
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fff;color:#3d4144;font:17px/1.85 -apple-system,BlinkMacSystemFont,'Segoe UI','Malgun Gothic',sans-serif;word-break:keep-all}a{color:#32789c;text-decoration:none}a:hover{text-decoration:underline}a:focus-visible,summary:focus-visible{outline:2px solid #32789c;outline-offset:4px}.masthead{border-bottom:1px solid #eee}.masthead>div{max-width:1160px;margin:auto;display:flex;align-items:center;padding:23px 30px;gap:30px}.brand{font-size:20px;font-weight:750;color:#3d4144}.brand small{font-size:12px;display:block;font-weight:400;color:#686e72}.masthead nav{margin-left:auto;display:flex;gap:27px;font-size:15px}.masthead nav a{color:#60686d}.layout{max-width:1160px;margin:40px auto 70px;display:grid;grid-template-columns:175px minmax(0,790px);gap:45px;padding:0 30px}.author{font-size:13px;color:#686e72;padding-top:7px}.author b{font-size:18px;color:#3d4144}.author p{margin:9px 0}.author nav{border-top:1px solid #ddd;margin-top:28px;padding-top:15px}.author nav a{display:block;margin:9px 0;color:#686e72}.review{font-size:12px;color:#7e684a;background:#faf6ed;padding:9px 13px;margin:0 0 26px;border-left:3px solid #d5b47a}h1{font-size:34px;line-height:1.45;letter-spacing:-1.2px;margin:0 0 14px;color:#303539}.meta{color:#757c80;font-size:12px;margin-bottom:32px}h2{font-size:25px;line-height:1.5;margin:54px 0 22px;padding-bottom:9px;border-bottom:1px solid #eee;letter-spacing:-.5px;color:#33383d}h3{font-size:20px;margin:33px 0 15px}p{margin:0 0 21px}strong{font-weight:700}hr{border:0;border-top:1px solid #eee;margin:35px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7;margin:22px 0 28px}td,th{padding:12px 13px;border-bottom:1px solid #e4e7e9;text-align:left}th{background:#f3f5f6}td:first-child{width:35%}pre{background:#f5f6f7;border:1px solid #e9ecee;padding:20px;overflow-x:auto;font:13px/1.8 Consolas,monospace;border-radius:3px}code{font:13px/1.65 Consolas,monospace}p code,td code{background:#f1f3f4;padding:2px 4px}img{display:block;width:100%;height:auto;margin:28px auto 10px}.caption{font-size:12px;color:#6c7379;line-height:1.7;margin:9px 0 25px}.flow{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;border:1px solid #dce2e6;background:#fafbfc;padding:24px 18px;margin:28px 0}.flow>div{flex:1;text-align:center}.flow b{display:block;font-size:16px}.flow span{font-size:12px;color:#677681}.flow i{font-style:normal;color:#84929b}.flow p{width:100%;border-top:1px solid #e0e5e8;padding-top:14px;margin:15px 0 0;text-align:center;font-size:12px;color:#5b6a75}details{margin:25px 0;padding:14px 18px;border:1px solid #dfe5e9;font-size:14px;background:#fafbfc}summary{cursor:pointer;font-weight:600}details p{margin:15px 0 0}.formula{font:14px/2 Consolas,'Malgun Gothic',sans-serif;overflow-x:auto;padding:18px 0 0;white-space:nowrap}.quota{display:flex;margin:26px 0;color:white;font-size:14px;line-height:46px}.quota span:first-child{width:75%;background:#477da8;padding-left:15px}.quota span:last-child{width:25%;background:#b5643a;text-align:center}footer{border-top:1px solid #eee;padding:25px 30px;background:#f4f5f6;color:#757c80;font-size:12px}footer>div{max-width:1100px;margin:auto}@media(max-width:800px){.layout{display:block;padding:0 20px;margin-top:25px}.author{display:none}.masthead>div{padding:17px 20px}.masthead nav{gap:17px;font-size:13px}.brand{font-size:17px}h1{font-size:28px}body{font-size:16px}h2{font-size:22px;margin-top:40px}h3{font-size:19px}td,th{padding:9px 7px}table{font-size:13px}pre{padding:14px;font-size:12px}.flow{padding:18px 9px;gap:4px}.flow b{font-size:13px}.flow span{font-size:10px}.flow p{font-size:11px}.review{font-size:11px}img{margin-top:20px}}@media(max-width:410px){.masthead nav a:not(:last-child){display:none}}@media print{.author,.review,.masthead nav{display:none}.layout{display:block;margin:0;padding:0}h1{font-size:27px}body{font-size:12px}img{max-height:320px;object-fit:contain}details{break-inside:avoid}}
"""
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>하스스톤 전장 AI 만들기(1) - 검토용 초안</title><style>{style}</style></head><body>
<header class="masthead"><div><a class="brand" href="https://kimmusic.github.io/">Who am I<small>I don't know</small></a><nav aria-label="블로그"><a href="https://kimmusic.github.io/ps">PS</a><a href="https://kimmusic.github.io/daily">일기</a><a href="https://kimmusic.github.io/study">공부</a><a href="https://github.com/KIMMUSIC/hearthstone-battleground">GitHub</a></nav></div></header>
<div class="layout"><aside class="author"><b>KIMMUSIC</b><p>프로젝트 / 하스스톤 전장 AI</p><p><a href="https://github.com/KIMMUSIC">GitHub</a></p><nav aria-label="초안 파일"><a href="draft.md">Markdown 원문</a><a href="sources.md">근거·편집 메모</a></nav></aside>
<main><div class="review">수정 초안 · 미게시 · 검토용 미리보기</div><h1>하스스톤 전장 AI 만들기(1)<br/>- 학습 환경 다시 만들기</h1><div class="meta">2026.09.06 · project</div><article>{article}</article></main></div><footer><div>© KIM MUSIC · 기존 블로그 형식에 맞춘 로컬 미리보기</div></footer></body></html>"""
(ROOT / "index.html").write_text(html, encoding="utf-8")
print("Rendered draft.md and four SVG/PNG figures. No Git or publication operations.")
