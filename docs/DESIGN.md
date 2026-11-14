# Design — Agentic Distillation on τ²-bench retail

_Stage 5, approved 2026-11-13. Decisions are in DECISIONS.md (D1–D3). Requirement IDs (O/U/P/R/AC) refer to REQUIREMENTS.md._

## Summary
- **Approach:** behaviour cloning. The **teacher** is DeepSeek (non-thinking mode, short visible thoughts). The **student** is **Tiny Aya En-Thinker** (3.35B, 32K context, run in `/no_think` mode), trained with LoRA.
- **Environment:** **τ²-bench retail**.
- **User simulator:** DeepSeek Flash for every run.
- **Optional:** one DAgger-lite round.

## Flow
```
τ²-bench retail (pinned commit)  ──(C1) split: train 60 · dev 14 · test 40
        │
(C2) Teacher runner [Mac]: DeepSeek agent + DeepSeek-Flash user sim, K=6 per train task ──(C8) ledger
        │
(C3) Raw trace store (all episodes, incl. failures) ───────────────────────► HF dataset (C11)
        │
(C4) Filter (reward=1) + formatter (render → tokenize → assistant-only labels; over-length rejected)
        │
(C5) Trainer [Kaggle T4]: TRL SFT + LoRA, fp16, checkpoints ──► adapter ──► HF model (C11)
        │
(C6) Student agent adapter (τ² HalfDuplexAgent): render → generate (vLLM | HF) → parse
        │
(C7) Eval harness: B0 · S2 · teacher, same user sim, dev then test, k=3 ──► records ──► metrics + cost (C8)
        │                                                         └─► (C9) threshold gate
(C10, optional) DAgger-lite: S2 on train tasks → teacher relabels up to the first error → retrain → C7
```

## Components
| # | Component | Responsibility | Runs on |
|---|---|---|---|
| C1 | Task split | Fixed IDs with a sha256 hash. Dev comes out of the official train set; the test set is run only for final reporting | Mac |
| C2 | Teacher runner | τ² `LLMAgent` with DeepSeek (non-thinking, temperature > 0) against the DeepSeek-Flash user simulator, K runs per train task | Mac (API only) |
| C3 | Raw trace store | τ² simulation JSON plus metadata: model, mode, trial, usage, cost, reward breakdown | Mac → HF |
| C4 | Filter + formatter | Keep reward = 1; render to the student protocol; tokenize; set labels to −100 on everything except assistant tokens; **reject and count over-length episodes, never truncate** (R1) | Mac / Kaggle |
| C5 | Trainer | TRL `SFTTrainer` on pre-tokenized data. LoRA, fp16 (R7), gradient checkpointing, explicit `max_length`, batch size 1 with gradient accumulation, epoch checkpoints, resume support | Kaggle/Colab T4 |
| C6 | Student agent adapter | τ² `HalfDuplexAgent`: render state → generate → parse. Identical for B0 and S2 | Kaggle/Colab T4 |
| C7 | Eval harness | Same orchestrator, user simulator and tasks for every model (P1). Reward = DB check × substring COMMUNICATE scorer (P2, R5) | Mac (teacher) / T4 (students) |
| C8 | Cost ledger + budget guard | Per-call token usage (cache hit / miss / output), split agent vs user simulator. Hard stop when the budget slice is spent. Student GPU wall time | Both |
| C9 | Threshold registry | `results/threshold.md`, written and dated after the baselines and before the S2 test run (P4, AC10) | — |
| C10 | DAgger-lite (optional) | Relabel failed student episodes on **train tasks only**, up to the first invalid call or tool error. One more SFT round from the S2 adapter | Mac + T4 |
| C11 | Publishing | Public HF model and dataset collections, GitHub repo, write-up (AC1–AC3, AC8, AC9) | — |

## Interfaces
1. **Split file:** `{train_ids[60], dev_ids[14], test_ids[40], tau2_commit, sha256}`
2. **Raw episode record:** the τ² simulation plus `{model, mode, trial, user_sim_model, usage{hit, miss, out}, cost_usd, reward, reward_breakdown}`
3. **Formatter, the single source of truth for training and inference:**
   - `render(policy, tools, messages) → tokens`
   - `parse(text) → ToolCall | Text | ParseError`
   - **Protocol:** the assistant writes `<tool_call>{"name":…,"arguments":{…}}</tool_call>` or plain text. Tool results come back in a user turn as `<tool_result name=…>…</tool_result>`, since no Aya template has a tool role (R4).
   - Assistant turns use the En-Thinker response block, with an empty thinking block (`/no_think`).
4. **SFT example:** `{input_ids, labels, source_episode_id, n_tokens}`
5. **Eval record:** `{model, task_id, trial, reward, n_steps, n_tool_calls, n_invalid_calls, terminated_by (done | max_steps | loop | error), tokens, wall_s, cost_usd}`
6. **Metrics:**
   - pass^1 and pass^3
   - valid-call rate
   - steps per task
   - loop/abort rate: max steps reached, or the same call repeated 3 or more times
   - cost per successful task
   - bootstrap CIs over tasks, with a paired B0-vs-S2 comparison (AuditKIT if it accepts precomputed records)

## Behavioural rules
- **Invalid call:** send one `<tool_result>` error back. A second invalid call in a row ends the episode as an abort. Every invalid call is counted.
- **Dev use:** dev results drive checkpoint selection (2–3 checkpoints at most) and freeze prompt and generation settings. Dev tasks are never used for teacher data.
- **Test use:** B0 and the teacher can run on test before the threshold is written. S2 runs on test only after `threshold.md` exists.

## Failure paths
| Failure | Priority | Detection → handling |
|---|---|---|
| Train/inference format mismatch | P0 | A render→parse round-trip over all teacher episodes must reproduce every call |
| Loss-mask bug | P0 | Mask-coverage stats plus decoding a sample of unmasked tokens; fix before training |
| Over-length episode | P2 (32K student) | Length histogram; exclude and report the count |
| Test contamination | P0 | Split hash; generation and DAgger-lite read train IDs only |
| Teacher success rate too low | P1 | Raise K within budget, or add airline train tasks |
| Budget overrun | P1 | Ledger hard stop; reduce K or k |
| User-simulator noise | P1 | Same simulator for every model; CIs; spot-check failures |
| Unparseable base-student output | P1 | Counted as invalid calls; one error feedback |
| T4 out of memory | P1 | Lower `max_length`, drop the longest episodes, or use QLoRA |
| Session timeout | P1 | Checkpoints and resume |
| API rate limits / non-determinism | P2 | Backoff and retries; log seeds and temperatures |

## Assumptions to verify early
- vLLM runs Cohere2 + LoRA on a Kaggle T4.
- A T4 fits the needed sequence lengths.
- The substring COMMUNICATE scorer is adequate.
- DeepSeek works through LiteLLM as both agent and user.
- AuditKIT accepts precomputed records.
- Cost is about $0.006 per episode.

## S2a results (2026-11-14)
- Formatter implemented per D5 and D6; tests cover template equality, parse round-trip (246 turns, 0 mismatches) and the token-level prefix property.
- Fixed system prefix: 4,632 tokens (6.7 KB policy, **16** tool schemas; earlier docs said 17, which was not verified).
- Rendered teacher episodes: median ~9.5K tokens, max 10,354; none near the 32K limit.
- Still untested: whether the real model can follow the dialect (S2b), and the loss masks (S4).

## S2b results (2026-11-14)
- Verified on Kaggle (2×T4, Python 3.13.16): install, model load, a real tau2 episode against the DeepSeek user simulator, eval record, ledger. Secrets are available only in UI-triggered runs, not CLI-pushed versions.
- B0 (strict parser) aborted on turn 1: valid call format, but hallucinated tool results. Led to D7. Throughput ~12 output tokens/s; per-episode time is an extrapolation until measured.
