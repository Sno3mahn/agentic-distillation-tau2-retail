# Domain Glossary — Agentic Distillation

_Stage 3 (domain learning). Created 2026-11-12. Definitions only, no decisions; see REQUIREMENTS.md for those._

**Evidence tags:** [V] verified from a primary source on 2026-11-12 · [M] well known, cited from memory and not re-checked · [C] unverified community claim · [A] arithmetic, not a measurement

## A. What "distillation" can mean
- **Agent / trajectory:** an agent is an LLM looping through observe → act (often a tool call) → observe. A trajectory is the full record of one episode: the task, each assistant turn, each tool call and result, and the outcome.
- **Logit (token-level) KD (knowledge distillation):** the student matches the teacher's probability distribution over the next token at every position. It needs the teacher's per-token probabilities over a shared vocabulary. *Impractical here:* DeepSeek and Aya use different tokenizers [M], and DeepSeek's docs don't mention logprobs [V].
- **Sequence-level KD / behaviour cloning:** the teacher generates complete trajectories, and the student is fine-tuned with SFT (supervised fine-tuning) to reproduce the teacher's turns. Introduced as sequence-level KD by Kim & Rush (2016) [M]. AgentTune defines its `distill` stage as exactly this [V]. Applied to tool-using agents with 0.5–3B students by Kang et al., NeurIPS 2025 [V]: [arXiv 2505.17612](https://arxiv.org/abs/2505.17612).
- **Compounding error (exposure bias):** a cloned student is only trained on states the teacher reached. At run time its own mistakes lead it into unfamiliar states, and the errors pile up. DAgger (Ross et al., 2011) showed that training on the learner's own states reduces this [M].
- **On-policy distillation:** the student generates its own trajectories and the teacher supplies feedback on them. GKD (Agarwal et al., ICLR 2024) is the standard form, and it uses token-level probabilities [V]: [arXiv 2306.13649](https://arxiv.org/abs/2306.13649). A text-only variant has the teacher write the next correct action for the student's partial trajectory [M].
- **DAgger-lite (one round of on-policy correction):** a single, simplified round of DAgger, added *after* a first behaviour-cloning run:
  1. run the trained student on the training tasks
  2. at the points where it goes wrong, ask the teacher for the correct next action from that exact state
  3. add those corrections to the data and fine-tune once more, on the same model and adapter
  - It targets compounding error at the cost of extra teacher calls plus a second training and evaluation round [M; the term "DAgger-lite" is our shorthand].
- **RL (reinforcement learning), for contrast:** the student learns from reward alone, with no teacher. It's often stacked after SFT [M]. AgentTune is mainly RL (GRPO, PPO, DPO, RLOO, BCO) [V].
- **GRPO (Group Relative Policy Optimization):** an RL method that samples several attempts at the same task and scores each against the group's average reward, then pushes the model toward the above-average attempts. It needs no separate value model, but it needs many rollouts on the GPU [M].

## B. Trajectory mechanics
- **Chat template:** the Jinja program that turns a list of messages into the model's token string. It must be identical at training and inference. For Aya models, tool support in the template is unverified [V that it's undocumented; C that Tiny Aya was not trained on tools].
- **Tool role / tool schema:** the JSON description of each function, and the message role (`tool`) that carries results back to the model. TRL's SFT accepts a `tools` column plus `tool_calls` and `tool` messages [V]: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer).
- **Schema conversion:** rewriting teacher output (OpenAI-style `tool_calls` JSON) into the student's format. A common source of data bugs, such as double-encoded arguments [M].
- **Assistant-only loss (loss masking):** training loss is computed only on assistant tokens, not on prompts or tool results. In TRL this needs `{% generation %}` tags in the template, which TRL adds automatically only for known model families, or else pre-computed `assistant_masks` [V].
- **Success filtering / rejection sampling:** keep only teacher episodes that pass the automatic check, and optionally sample several attempts per task. It skews the data towards easy tasks and can drop error-recovery examples. STaR (Zelikman et al., 2022) is a well-known example [M].
- **Reasoning traces:** whether training targets include the teacher's thinking. Options are none, short ReAct-style thoughts (Yao et al., ICLR 2023) [M], or long thinking. DeepSeek's API has thinking mode on by default [V].
- **Dual-mode thinking (`/think`, `/no_think`):** the Tiny Aya thinker models can reason before answering or skip it.
  - `apply_chat_template(..., enable_thinking=True/False)` appends `/think` or `/no_think` to the user turn.
  - Reasoning appears between `<|START_THINKING|>` and `<|END_THINKING|>`, and the answer between `<|START_RESPONSE|>` and `<|END_RESPONSE|>`.
  - In no-think mode the template inserts an empty thinking block [V]: [En-Thinker card](https://huggingface.co/CohereLabs/tiny-aya-en-thinker).
- **Context budget / truncation:** both Aya models have 8K context [V]. TRL's SFT `max_length` defaults to 1024 with `keep_start` truncation [V], which silently drops later turns.

## C. Training on free-tier GPUs
- **Full fine-tuning memory:** about 16 bytes per parameter with Adam [M], so about 54 GB for 3.35B parameters [A]. Infeasible on a 16 GB T4.
- **LoRA (low-rank adaptation):** freeze the base model and train a small low-rank update ΔW = B·A on selected layers, usually under 1% of the parameters (Hu et al., 2021) [M]. The output is a small adapter.
- **QLoRA / NF4:** LoRA on top of a base model stored in 4-bit NormalFloat, with double quantization and paged optimizers. The authors claim parity with 16-bit fine-tuning [V]: [arXiv 2305.14314](https://arxiv.org/abs/2305.14314). Rough base sizes: Tiny Aya ~2 GB, 8B ~5 GB [A].
- **Activation memory / gradient checkpointing:** activation memory grows with sequence length × batch size. Gradient checkpointing recomputes activations instead of storing them, and TRL enables it by default [V].
- **fp16 vs bf16:** T4/P100 GPUs lack bf16 [M]. TRL defaults to bf16 unless fp16 is set [V]. FlashAttention-2 is unsupported on T4 [M], and TRL's `padding_free` requires it [V].

## D. Evaluation
- **Static vs interactive evaluation:** static scores the next call given a fixed correct history, as in BFCL (AST checks) [V]: [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html). Interactive runs the whole loop and scores the outcome, as in τ-bench [V]. Only interactive evaluation exposes compounding errors.
- **Verification types:**
  - string or AST match: brittle to equivalent calls
  - execution-based: run the call and compare results
  - final-state: compare the database or environment end state
  - LLM judge: noisy and costly
- **τ-bench / τ²-bench:**
  - τ-bench scores by final database state: [arXiv 2406.12045](https://arxiv.org/abs/2406.12045) [V].
  - τ²-bench is MIT-licensed, with 50 frozen tasks per domain in `text-v1`. Its reward is the product of DB × COMMUNICATE by default, and COMMUNICATE uses an LLM judge unless the substring scorer is chosen [V]: [eval docs](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md).
- **User simulator:** an LLM that plays the customer in conversational benchmarks [V]. It adds API cost and run-to-run variance, and its own mistakes count against the agent [M].
- **pass@k vs pass^k:** pass@k means at least one of k attempts succeeds (capability); pass^k means all k succeed (reliability). In 2024, gpt-4o succeeded on under 50% of τ-bench tasks, and retail pass^8 was under 25% [V].
- **Evaluation noise:** with 50 tasks at about 50% success, the standard error is about ±7 points [M/A]. AuditKIT offers bootstrap significance tests [V].
- **Contamination / dev–test split:** the model has effectively seen the test tasks, through exact or near-duplicates, public data containing the benchmark, or checkpoint selection on the test set. The usual fix is a separate dev split for tuning [M].

## E. Cost and licensing
- **Cost per successful task:** total run cost ÷ number of successes.
  - Teacher: measured from DeepSeek's per-request `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` and output tokens [V]: [caching docs](https://api-docs.deepseek.com/guides/kv_cache).
  - Student: GPU $/hour ÷ successful tasks per hour.
- **Utilization / batching:** a self-hosted GPU costs the same idle or busy. Serving many requests at once (e.g. vLLM) raises throughput several-fold [M].
- **Context caching:** DeepSeek's caching is on by default, matches by prefix and is best-effort. Off-peak, cached input costs $0.003/M vs $0.15/M uncached [V]: [pricing](https://api-docs.deepseek.com/quick_start/pricing).
- **Peak pricing:** DeepSeek's off-peak rate is half price. Peak is 01–04 and 06–10 UTC on weekdays [V], i.e. 06:30–09:30 and 11:30–15:30 IST [A].
- **HF GPU prices:** T4-small Space $0.40/h, T4 Endpoint $0.50/h, L4 $0.80/h. ZeroGPU is free but quota-limited [V]: [HF pricing](https://huggingface.co/pricing).
- **CC-BY-NC-4.0:** sharing and adapting are allowed with attribution. Use "primarily intended for commercial advantage or monetary compensation" is not [V]: [deed](https://creativecommons.org/licenses/by-nc/4.0/). Both Aya models are under it, so fine-tuned derivatives inherit it [V]: [Tiny Aya card](https://huggingface.co/CohereLabs/tiny-aya-global).
- **Dataset terms:** each source's terms must allow training and redistribution (AC9).
  - DeepSeek API outputs: rights assigned to the user; distillation permitted [V].
  - APIGen-MT-5k: CC-BY-NC, research only, OpenAI clause [V]: [card](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k).
  - xLAM-60k: CC-BY-4.0 with a "research only" note [V]: [card](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).
