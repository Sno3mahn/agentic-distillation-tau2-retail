# Experiment Log

_Chronological. Numbers here come from saved records under `data/` (gitignored); commands are given so they can be regenerated._

## E1 — Teacher pilot, three variants (2026-11-14, slice S3)
**Question:** how often does the DeepSeek teacher succeed on retail train tasks, what does each variant cost, and which variant should generate the training data?

**Setup:** 10 train tasks (seeded sample, seed 20261114: 4 11 19 30 63 66 67 73 78 93) × 2 trials × 3 variants = 60 episodes. `deepseek-flash` agent and user simulator; user simulator temperature 0; trial seeds 300/301 shared across variants (paired). Reward = DB × substring COMMUNICATE (D4). Run off-peak (Saturday). Regenerate: `python -m agentic_distill.metrics base=data/raw/s3_pilot10_base rationale=data/raw/s3_pilot10_rationale thinking=data/raw/s3_pilot10_thinking`.
Variants: **base** non-thinking; **rationale** non-thinking + one-sentence rationale before each tool call; **thinking** DeepSeek thinking mode (reasoning not shown to the student).

| metric | base | rationale | thinking |
|---|---|---|---|
| episodes passed | 17/20 | 19/20 | 20/20 |
| pass^1 (95% CI over tasks) | 85% [70, 100] | 95% [85, 100] | 100% [100, 100] |
| pass^2 | 70% [40, 100] | 90% [70, 100] | 100% [100, 100] |
| valid-call rate | 96% | 96% | 97% |
| steps per episode | 12.7 | 12.9 | 12.4 |
| loop/abort rate | 0% | 0% | 0% |
| cost per episode / per success | $0.0020 / $0.0023 | $0.0020 / $0.0021 | $0.0031 / $0.0031 |
| failure causes | 2 COMMUNICATE, 1 DB | 1 COMMUNICATE | none |

Paired differences vs base (tasks in common, 95% CI): rationale pass^1 +10 pts [0, 25]; thinking pass^1 +15 pts [0, 30].
Rendered token lengths (student format): median ~9.0–9.2K, p90 ≤ 10.7K, max 10,851; none near the 32K limit. Combined with the earlier 2-task retries (base 5/8 including the first smoke, rationale 5/6, thinking 7/7 + 1/1) the direction is the same.

**Findings**
- Teacher success is high on these tasks (85–100%), so data yield is not a bottleneck. Every sampled task had at least one success in every variant.
- Differences between variants are not statistically distinguishable at this size: the paired CIs touch 0.
- Most failures were COMMUNICATE (the agent did not state a required figure), concentrated in tasks 4 and 19, not DB errors.
- Thinking costs ~1.55× per episode; 20/20 thinking episodes recorded reasoning tokens and 0/40 of the others did, so the mode switch works.
- Spend this experiment: ~$0.14. Slice `s3_pilot` total $0.1945 of $0.40; project total ≈ $0.20 of $5.

**Limitations:** 10 tasks, 2 trials (pass^3 not computable); single seed pair; substring scoring can give false passes (never false failures); user-simulator noise; a seeded sample of train tasks, not a ranking over the full set.

**Not decided here:** K, k, the variant for S4 data. See DECISIONS.md once chosen.

## E0 — B0 (base student) smoke runs (2026-11-14)
See DESIGN.md "S2b results" and "B0 under D7". Qualitative only (3 dev episodes, all reward 0).

## E2 — Teacher data generation and hard-task rerun (2026-11-14, slice S4)
**Question:** how well does the teacher do on all 60 train tasks, why does it fail, and does a second, thinking-mode pass on the hard tasks add useful training data?

**Setup**
- **Pass 1 (`rationale` variant):** 60 train tasks × K=6 = 360 episodes, non-thinking + one-sentence rationale, off-peak. Runs `data/raw/s4_rationale_p0..p5`. Cost ≈ $0.74.
- **Pass 2 (`thinking_rationale` variant, added to `configs/run.yaml`):** the 17 tasks with ≤4/6 successes in pass 1, excluding task 59, × K=8 = 136 episodes. Runs `data/raw/s4_thinkrat_h0..h5`. Cost $0.583. Slice `s4_teacher_data` total $1.32 of $3.00.
- Reward = DB × substring COMMUNICATE (D4). User simulator DeepSeek Flash, temperature 0.

**Results**
- Pass 1 over the 60 train tasks: pass^1 82% [75, 88], pass^2 72%, pass^3 65% [55, 75]. 94 of the 496 episodes in both passes failed (reward < 1).
- Pass 2: 107/136 = 78.7% success on the 17 tasks. In pass 1 the same 17 tasks scored 52/102 = 51%. **This is not a clean estimate of the thinking benefit:** the tasks were selected because pass 1 did badly on them, so their pass-1 rate is biased low (regression to the mean). The variant change and the selection effect are not separated.
- Largest per-task changes (pass 1 → pass 2): task 20 1/6 → 7/8, task 107 2/6 → 8/8, task 44 2/6 → 8/8. Still weak: tasks 104 (2/8) and 105 (2/6), and 41 (4/8).

**Why the teacher failed (pass 1; 53 DB mismatches, classified by comparing its successful write calls with the gold writes)**
- 29 (55%): right tool, wrong arguments (wrong replacement item in 16 writes; wrong order in 14, 12 of them from task 59; wrong payment method in 4).
- 14: only some required writes made (tasks 41, 104). 7: no write at all. 3: extra or wrong-type writes.
- **Task 59 (0/6) is a user-simulator/gold ambiguity, not an agent error:** orders carry no date; the simulator asked to cancel "the older one" and named #W2702727, while gold cancels #W8268610. The agent followed the explicit request in 6/6 runs. Excluded from the rerun; no kept examples for it (59/60 tasks covered).
- **Task 41** is partly underspecified (the scenario never names a payment method; gold uses PayPal).
- 12 COMMUNICATE failures: in all 12 the required figure does not appear in the agent's text (crude check), so they look like omissions, not formatting mismatches.

**Dataset:** `data/sft/train_v2.jsonl` = 370 examples (287 `rationale`, 83 `thinking_rationale`), 59/60 tasks, 5–8 per task (cap 8, seeded shuffle, 32 over-cap episodes dropped), none over length. Median 8.8K tokens, p90 11.3K, max 12.9K; 3.34M tokens total, of which 430K (12.9%) carry loss. The earlier 295-example `train.jsonl` is kept unchanged. Regenerate: `python -m agentic_distill.build_sft "data/raw/s4_rationale_p*" "data/raw/s4_thinkrat_h*" --cap-per-task 8 --out data/sft/train_v2.jsonl`.

**Comparison with published numbers (context, not a benchmark):** a third-party leaderboard submission (PR #709, unverified, no stated thinking mode) reports V4.1-Flash retail pass^1 88.6%, pass^3 78.7% on all 114 tasks with the official reward and 4 trials. Ours is lower (82% / 65%) but differs in tasks (60 train), reward (substring, not the NL judge), prompt (rationale instruction) and trials (6). Not comparable; the size of each factor is unknown.

**Limitations**
- Substring scoring can pass an episode that states the figure wrongly; a semantic audit of the kept episodes was considered and **not done** (user decision), so false passes among the 370 are unmeasured.
- 83 examples come from a different generator (thinking mode, visible reasoning dropped), so the data mixes two teacher behaviours. Tagged by `variant` for ablation.
- The user simulator is noisy (task 59). The same noise applies to the student's dev/test evaluation.
- Pass 2's 17 tasks were chosen from the same data they are compared against.

## E3 — LoRA SFT of the student (2026-11-15, slice S5)
**Setup:** Tiny Aya En-Thinker (3.35B), LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down; `data/sft/train_v2.jsonl` (370 examples, none dropped for length); 3 epochs, lr 1e-4 with 5% warmup, grad-accum 8 (47 optimizer steps per epoch), fp16 with GradScaler, fp32 LoRA weights, gradient checkpointing, seed 20261115. One Kaggle T4. Loss only on assistant turns, with `lm_head` applied only at supervised positions (`selective_loss`, tested equal to the HF loss and gradients in `tests/test_loss_positions.py`).
**Memory fixes needed to fit 12.9K-token examples on a 15 GB T4:** chunked selective loss (the full 262K-vocab logits do not fit); `use_gqa_in_sdpa` patched off so SDPA uses the memory-efficient kernel instead of falling back to the T×T math kernel (`disable_sdpa_gqa`, output equality tested).
**Results** (`data/kaggle_out_s5i/out/lora/train_log.jsonl`): mean train loss per epoch 0.629 / 0.437 / 0.381; 0 skipped steps; peak memory 13.4 GB; median 397 tokens/s. Checkpoints `epoch1..3`.
**Limitation:** training loss only; no held-out loss was computed. Generalisation is judged by the dev evaluation (E4).

## E4 — Dev evaluation and checkpoint choice (2026-11-15, slice S6)
**Setup:** epoch2 and epoch3 on the 14 dev tasks × 1 trial, greedy, max_new_tokens 1024, user simulator DeepSeek Flash temperature 0. Kaggle T4 x2 (one checkpoint per GPU).
**Results** (12 tasks scored; tasks 103 and 110 crashed with CUDA out-of-memory before per-episode crash records existed, so they are missing for both checkpoints):

| metric | epoch2 | epoch3 |
|---|---|---|
| pass^1 | 25% [0, 50] | 42% [17, 67] |
| valid-call rate | 87% | 84% |
| loop/abort rate | 33% | 0% |
| steps | 18.5 | 18.5 |

Paired epoch3 − epoch2: pass^1 +17 pts [−17, +58] (not significant). **epoch3 chosen** (higher point estimate, no aborts); see D9.
**Limitations:** 12 episodes, 1 trial; the difference is within noise. The OOM cause on long episodes is not confirmed (hypothesis: KV cache plus prefill on 30K+-token contexts).

## E5 — Test evaluation (2026-11-15, slice S7; in progress)
Pre-registered in `results/threshold.md` (written 13:49 IST, before any student test episode): frozen epoch3 adapter (sha256 `88544be6…c85b1`), 40 test tasks × 3 trials, crashes and aborts score 0, bar = paired 95% task-bootstrap CI lower bound of (student − B0) pass^1 above 0.

**Teacher (`base` variant, context only), 40 tasks × 3 trials:** pass^1 89% [83, 95], pass^3 72% [58, 85], valid-call rate 98%, 12.3 steps, cost per success $0.0023. Cost $0.25.

**Student, how the test set was run:** attempt 1 on Kaggle split the tasks over two GPUs; GPU 0 died on a run-name collision (fixed: pid suffix), so only the "odd" half (`test_ids[1::2]`, 20 tasks) ran. The "even" half runs on Colab (single T4, `notebooks/s7b_test_eval_colab.ipynb`, same code at commit 341ab2a, same adapter hash; library versions in `results/env_colab.json`). See D11.

**Student, odd half (20 tasks × 3 = 60 episodes):** pass^1 42% [27, 57], pass^3 15% [0, 35]; valid-call rate 92%; 17.1 steps; loop/abort 8%; 1 crash (task 36, CUDA OOM at the first turn), 2 `error` terminations (tasks 39, 53), 1 `max_steps` (task 111). 10 of 20 tasks never passed. Same 20 tasks, teacher − student pass^1: 50 pts [35, 65].
**Pending:** even half (Colab), B0 on all 40 tasks (Kaggle `agentic-distill-s7-b0-test-eval`), then the pooled paired comparison against B0 and a sensitivity check excluding crashes.
**Student cost per success ($0.0015)** counts only the user-simulator API calls, not GPU time; the GPU cost is added in the write-up.

## E6 — Continued training (2026-11-15, post-hoc, not pre-registered)
**Question:** does more training beyond epoch3 help? Only for the write-up; the pre-registered test result stays epoch3 (D12).
**Setup:** start from the epoch3 adapter (`--init-adapter`), 2 more epochs, lr 5e-5, seed 20261116, same data. Kaggle T4.
**Results:** mean train loss 0.372 / 0.329 (cumulative epochs 4 / 5); 0 skipped steps. Checkpoints in `data/kaggle_out_s8/out/lora_ext/epoch{1,2}`.
**Pending:** dev evaluation on the first 8 dev tasks × 1 trial (`agentic-distill-s8-dev-eval`), compared against epoch3 on the same 8 tasks (4/8 in E4). With 8 episodes this can only show a large change.
