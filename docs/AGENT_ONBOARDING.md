# Agent onboarding — Control Channel Experiments / Sinii replication on DeltaAI

You're an AI coding agent helping replicate Sinii et al. 2025 ("Steering LLM Reasoning Through Bias-Only Adaptation," arxiv 2505.18706) on the NCSA DeltaAI cluster. This is Step 1 of a longer unpromptability experiment — the foundation everything else builds on.

Read this whole doc once before doing anything. After that, switch to your assigned Linear issue.

## Quick orient

- **Project tracker:** Linear workspace `Apto-parable` (team key `APT`), project "Control Channel Experiments". Parent issue is **APT-41**; per-block child issues are **APT-42 through APT-48**.
- **Your work lives on:** the `deltaai-replicate` branch of `https://github.com/robertnward/steering-reasoning` (this repo). The branch already contains the patched config, the bias patcher, and the sbatch script.
- **Cluster:** NCSA DeltaAI. Each node has 4× NVIDIA GH200 Grace Hopper Superchips (Hopper GPU 96GB HBM3e + 72-core ARM Grace CPU, NVLink-C2C). Partition `ghx4`, account flag `bgye-dtai-gh`. Allocation is 3,000 GH200-hours, shared.
- **Architecture:** Grace CPUs are **aarch64**, not x86_64. Most x86 pip wheels won't install. Sinii's `pyproject.toml` is x86-locked; we install with version flex (see APT-44).
- **Goal of Step 1:** Reproduce Sinii's mean@8 GSM8K accuracy for Qwen2.5-1.5B activation steering, within ~2pp of the paper's Table 1.

## Repo layout on DeltaAI

After Block 1 + 2 you should have:

```
/projects/bgye/$USER/steering-reasoning/          # this repo, deltaai-replicate branch
├── configs/train/rl/qwen2.5-1.5b/gsm8k/
│   ├── steering.yml                              # upstream (do not modify)
│   └── steering-deltaai.yml                      # our patched config
├── bin/helpers/
│   ├── modify_bias_vllm.sh                       # upstream (Docker-path hardcoded; broken on DeltaAI)
│   └── modify_bias_sitepackages.sh               # our replacement; takes site-packages path as arg
├── scripts/
│   └── sinii-replicate.sbatch                    # full-training SLURM submission
└── logs/                                          # sbatch writes here

/work/nvme/bgye/$USER/                            # fast NVMe scratch (1TB, NOT backed up)
├── hf_cache/                                     # symlinked from ~/.cache/huggingface
└── runs/                                         # training checkpoints, steering vectors
```

## Work protocol

1. **Read your assigned issue's description in full** before executing anything. The body contains every command you need, the `Done when` criteria, and known failure modes.
2. **Run steps in order**. Each section is roughly a discrete substep; verify the substep succeeded before moving to the next.
3. **Comment on the Linear issue after each substep**, briefly: command run, what the output told you. This is the audit trail. Aim for 2–6 comments per block, not a wall of text.
4. **On failure, stop and diagnose**. Do not blind-retry — especially anything that consumes GPU time. Post a comment with: what failed, the relevant log/error, what you tried, what you think is the cause. Then wait for human guidance unless the issue body documents the exact failure mode and a recovery path.
5. **When all `Done when` criteria pass**, set the issue status to **Done** and post a final summary comment naming the key outputs (paths to artifacts, mean@8 numbers, etc.).

## Cluster guardrails (do not violate without explicit approval)

- **Never run heavy compute on the login node.** Use `salloc` for interactive sessions, `sbatch` for batch jobs.
- **Always release `salloc` sessions when finished** (`exit` from the allocated node). Holding a GH200 idle burns ~1 GH200-h per wall-clock hour.
- **Stay under ~10 GH200-h per single block** unless the issue explicitly budgets more. If you're approaching that ceiling, post a comment and pause.
- **Do not modify upstream Sinii files** — anything outside `configs/train/rl/.../steering-deltaai.yml`, `bin/helpers/modify_bias_sitepackages.sh`, `scripts/sinii-replicate.sbatch`, and the `logs/` dir is upstream. Patches go in our own files only.
- **Never `git push --force`**. Never push to anything other than the `deltaai-replicate` branch of `origin` (the personal fork).
- **Do not commit dataset downloads, model weights, or `wandb/` artifacts to git.** These belong in `/work/nvme/bgye/$USER/`.
- **Do not store secrets in the repo.** `WANDB_API_KEY`, `HF_TOKEN`, `WANDB_ENTITY` come from the submitting shell's environment.

## Linear integration

Three workflows in priority order — use whichever is available:

**(A) Linear MCP server.** If you have the Linear MCP tools (`save_issue`, `list_issues`, `save_comment`, etc.) available, use them directly. This is the cleanest path.

**(B) Linear CLI / `gh`-equivalent.** If a Linear CLI is configured, use it for fetch + update.

**(C) Inline issue body, no Linear access.** If neither (A) nor (B) is available, the issue body should have been pasted into your initial prompt by the human dispatcher. Execute the steps and **report progress back to the dispatcher** as ordinary chat messages — the dispatcher will mirror them onto Linear.

Whichever workflow: **comments go on the assigned child issue**, not the parent (APT-41). This keeps the audit trail organized per block.

## When to stop and ask

- Any step failure not documented in the issue's risk ladder.
- Any ambiguity between the issue body and this onboarding doc (this doc is authoritative for cross-cutting concerns; the issue body is authoritative for that specific block's steps).
- A choice you'd need to make that affects more than just this block (e.g. "should I try a different vllm version?" — yes for known fallbacks listed in the issue; ask for anything else).
- Allocation budget concerns.
- If the upstream Sinii code looks like it has a bug — don't try to fix it, document it.

Post a comment on the issue with what's blocking and what you tried. Don't close, archive, or change status to anything other than what your guardrails allow.

## Pointers to deeper context (if you need it)

These docs live in the parent worktree, not in this repo. If your dispatcher has copied them to you, they're worth a skim:

- `docs/unpromptability-experimental-handoff.md` — the full experiment design and why Step 1 matters.
- `docs/sinii-repo-audit.md` — the audit that produced the patched config + sbatch.
- `docs/sinii-replication-runbook.md` — the runbook that the Linear issues were derived from.
- `docs/task-flowchart.md` — visual of how the eight blocks connect.

You don't need any of these to do your job — your issue body is self-contained. They're there if you hit something unexpected and need to understand the bigger picture.
