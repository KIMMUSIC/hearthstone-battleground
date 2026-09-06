# Local Execution Switch Plan

Status: implemented and verified on 2026-09-06. This records the transition decision; current progress is in DAILY_GOALS_2026-09.md and daily result documents.

## Requirements Summary

The active September work no longer waits for Grok Bot. Future runs continue from the blocked 9/9 opponent-diversity training stage on the user's PC. The local path uses WSL Ubuntu for Linux supervisor behavior, a dedicated Python 3.11 virtual environment for RL dependencies, CPU-only execution, the existing CPU1 and sampled RSS2GiB budget, isolated immutable execution copies, unique run IDs, and fresh evidence before each downstream date.

Historical Grok Bot reports remain valid as provenance for experiments 001-008 and for the 9/8 smoke/9/9 rejection trail. They are not the current execution authority.

## RALPLAN-DR Summary

Principles:
- Preserve experiment provenance and never rewrite past completion claims.
- Continue from the first unmet dependency, currently 9/9 training.
- Keep development checkout separate from each training execution copy.
- Prove environment and resource assumptions with fresh local checks.
- Prefer the smallest local campaign that satisfies the existing 009 specification before moving to 9/10.

Decision drivers:
- Grok Bot review rejected the fixed 9/9 command repeatedly, and no pending answer remains.
- This PC has WSL2 Ubuntu and enough D: space for the current small experiments.
- The repository already has a Linux supervisor, while the strict resource supervision path is Linux-oriented.

Viable options:
- Option A: Local WSL supervisor with a dedicated Python 3.11 venv. Pros: matches existing Linux supervisor behavior, keeps run isolation clear, avoids Grok review blocking. Cons: requires confirming or installing Python 3.11 inside WSL before full training.
- Option B: Windows PowerShell execution with the existing Python 3.11 packages. Pros: Python 3.11 and CPU Torch already load. Cons: `scripts/supervise.py` has Linux process-group and `/proc` assumptions, so resource/child-process evidence would be weaker.
- Option C: Wait for Grok Bot recovery. Pros: preserves the original remote handoff shape. Cons: no confirmed recovery path and the current task explicitly switches away from Grok Bot.

Chosen option: Option A.

## Acceptance Criteria

- `AGENTS.md` assigns current experiment execution to Codex/local PC instead of Grok Bot and requires WSL + dedicated venv gates before long training.
- `README.md` points new sessions to local execution/supervisor docs and marks Grok docs as historical.
- `docs/PLAN_2026-09.md` changes current and future execution rows from Grok Bot to local supervised execution.
- `docs/DAILY_GOALS_2026-09.md` changes the active 9/9-9/15 flow to local execution, with 9/9 still incomplete until six training runs and audit evidence exist.
- Completion claims require local artifacts and successful audits; both 9/9 training and 9/10 evaluation now meet that condition.
- Verification includes a fresh grep for remaining current-duty Grok wording and local WSL/Python evidence.

## Implementation Steps

1. Update `AGENTS.md` role ownership so Codex controls local implementation, local supervised training/evaluation, and result analysis. Keep execution-copy isolation.
2. Update `README.md` onboarding and role table for WSL local operation. Keep historical Grok reports discoverable.
3. Update `docs/PLAN_2026-09.md` to make "local supervised execution" the current execution lane and preserve earlier Grok history as past evidence.
4. Update `docs/DAILY_GOALS_2026-09.md` so the common prompt and daily tasks no longer instruct use of `grok-bot`; 9/9 resumes locally, then 9/10-15 continue in order.
5. Re-read changed sections, run text searches for stale operational Grok instructions, and confirm the local WSL/Python facts used by the plan.

## ADR

Decision: Use local WSL Ubuntu supervised execution for the active September experiment plan.

Drivers: Grok Bot has no pending result and repeatedly rejected the fixed 9/9 command; the local PC has WSL2 and ample D: disk; the existing supervisor is Linux-oriented.

Alternatives considered: Windows-only execution is acceptable for lightweight smoke checks but does not match the supervisor contract. Waiting for Grok Bot was rejected by the user's latest instruction and by lack of a confirmed recovery path.

Why chosen: WSL preserves the existing Linux supervisor semantics while keeping all source, outputs, hashes, and run IDs on the user's machine.

Consequences: Before full training, execution must confirm or create a WSL Python 3.11 venv and re-measure runtime/RSS locally. Past Grok documents stay in the archive and should not be deleted or silently reinterpreted.

Follow-ups: Resume at 9/9 with local preflight, then run the fixed 2-condition x 3-seed x 8192-step campaign only after the local execution gate passes. Run 9/10 evaluation only after the six 9/9 outputs and audit are present.

## Decision check

WSL preserves the existing Linux process model. The dedicated Python3.11.15 environment passed dependency checks and107 tests, followed by six training runs and2000 audited evaluation episodes. The evidence is recorded in LOCAL_EXECUTION.md and runs/diversity-009-local-r1/local-verification.json.

## Changelog

- Initial transition decision captured for the Grok-to-local execution switch.
