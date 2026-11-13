# Implementation Plan — vertical slices (approved 2026-11-13)

## Context
Stage 5 design was approved on 2026-11-13.
- Distillation: behaviour cloning from a DeepSeek teacher (non-thinking, short visible thoughts) into Tiny Aya En-Thinker (32K, `/no_think`, LoRA).
- Environment: τ²-bench retail. Split: 60 train / 14 dev / 40 test.
- DeepSeek Flash simulates the user. K=6 teacher episodes per train task; k=3 eval trials per task.
- Tool calls: JSON in tags. On an invalid call, feed back one error, then end the episode.
- Inference: vLLM, falling back to HF `generate`. AuditKIT for statistics only. DAgger-lite (up to the first error) is optional.

Hard deadline: Nov 15 23:59 AoE (Nov 16 17:29 IST). Capacity: solo, 15–35 h.
Run location (user-decided 2026-11-13): API-only jobs (teacher generation, teacher eval) run on the Mac. Everything that loads an Aya model runs on Kaggle/Colab T4.

Rules for every slice:
- Each slice needs its own approval before coding.
- Before each slice: list its files and dependencies. Every install, dataset/model download, HF push and API spend needs explicit approval.
- After each slice: report files changed, commands run, outputs, and checks passed or failed.

## Proposed repo layout (created in slice 1, not before)
- `pyproject.toml`, with dependencies pinned.
- `src/agentic_distill/` modules:
  - `splits.py` (C1)
  - `ledger.py` (C8)
  - `teacher_run.py` (C2/C3)
  - `formatter.py` (C4 render/parse)
  - `student_agent.py` (C6, a τ² `HalfDuplexAgent`)
  - `build_sft.py` (C4 filter, tokenize, masks)
  - `metrics.py` (C7 metrics and bootstrap)
  - `evaluate.py` (C7 runner)
- Notebooks: `notebooks/train_kaggle.ipynb` (C5) and `notebooks/eval_student_kaggle.ipynb` (C6/C7).
- Config: `configs/run.yaml` with split, K, k, model IDs, temperatures, budget.
- Tests: `tests/test_formatter.py`, `tests/test_masks.py`, `tests/test_metrics.py`, `tests/test_splits.py`.
- `data/` is gitignored and holds raw traces, SFT sets and eval records. `results/` holds `threshold.md` (C9) and reports.
- New dependencies, each approved when its slice starts: tau2-bench at a pinned git commit (brings in LiteLLM), pytest; on Kaggle only: transformers, trl, peft, datasets, vllm, huggingface_hub; later: auditkit.

## Essential slices (in order)

### S0: Record the design (docs only, ~0.5 h)
- **Outcome:** the approved design is captured in docs.
  - `docs/DESIGN.md` (new): components C1–C11, flow, interfaces, failure paths.
  - `docs/DECISIONS.md`: add D3 (OPEN-1…9 outcomes, and the run-location decision).
  - `docs/REQUIREMENTS.md`: mark R8 resolved, note the OpenRouter limits finding, amend U4 (API jobs run on the Mac).
  - `docs/IMPLEMENTATION_PLAN.md` (new): this plan.
- **Check:** the user reviews the diffs.
- **Risk:** none.

### S1: One teacher episode, end to end, on the Mac (~2–3 h)
- **Outcome:** one command runs DeepSeek as the agent and DeepSeek Flash as the user on 2–3 retail train tasks. It saves raw τ² traces with token usage and cost, and prints the reward with its DB and substring-COMMUNICATE breakdown.
- **Also:** write the split file (60/14/40, with a sha256 hash) and the ledger.
- **Depends on:** S0; a DeepSeek API key (env var); approval of the tau2-bench install.
- **Files:** pyproject, `splits.py`, `ledger.py`, `teacher_run.py`, `configs/run.yaml`, `tests/test_splits.py`.
- **Acceptance checks:**
  - The split file is disjoint, and its counts are 60/14/40.
  - The rewards are computed.
  - The ledger's cost matches the token counts DeepSeek reports, at verified prices.
  - Spend is under $0.10.
- **Risks:**
  - τ²'s Python version, or LiteLLM's model string for DeepSeek V4.1 Flash.
  - Whether the substring scorer can be configured from code.
  - Whether DeepSeek's non-thinking mode passes cleanly through LiteLLM.

### S2: Formatter and base-student (B0) episode on a T4 (~3–4 h)
- **Outcome:**
  - `formatter.render/parse` for the JSON-in-tags protocol.
  - A round-trip test over the S1 teacher traces.
  - A token-length report.
  - `student_agent.py` runs **B0 (Tiny Aya En-Thinker in `/no_think` mode, HF generate) on 1 dev task in a Kaggle notebook**. The episode either finishes, or ends by the rule: one error is fed back, then the episode ends. Invalid calls are counted.
- **Depends on:** S1 traces; Kaggle T4; HF_TOKEN in Kaggle secrets; approval to download the model on Kaggle.
- **Files:** `formatter.py`, `student_agent.py`, `tests/test_formatter.py`, `notebooks/eval_student_kaggle.ipynb`.
- **Acceptance checks:**
  - Round-trip render→parse reproduces 100% of the teacher calls.
  - B0 runs one full τ² episode and produces an eval record.
  - Token lengths are logged.
- **Risks:**
  - Running τ²-bench inside Kaggle (dependencies, network access).
  - The Cohere2 architecture in the installed transformers version.
  - Generation speed.

### S3: Pilot and metrics (decision gate, ~2 h)
- **Outcome:**
  - About 20 teacher episodes on train tasks.
  - `metrics.py` computes pass^1, pass^3, valid-call rate, steps, loop/abort rate and cost per success.
  - A short pilot report: real $/episode, teacher success rate, token-length percentiles.
- **Gate:** confirm or adjust K, k and the budget split with the user before S4.
- **Files:** `metrics.py`, `tests/test_metrics.py`, `evaluate.py` (record schema).
- **Acceptance checks:**
  - The metrics match hand-computed values on a toy fixture.
  - The pilot cost is within ±30% of the estimate, or the plan is revised.
- **Risks:** a low teacher success rate (fallback: raise K or add the airline train tasks), and cost overrun.

### S4: Full teacher data and SFT dataset (~2–3 h, mostly unattended)
- **Outcome:**
  - 60×6 teacher episodes (budget guard on), all traces stored.
  - `build_sft.py` filters to reward = 1, renders with the formatter, tokenizes, and builds −100 masks.
  - Over-length episodes are excluded and counted, never truncated.
  - Statistics: number kept, tasks covered, mask coverage, length histogram.
- **Depends on:** S3 gate.
- **Files:** `build_sft.py`, `tests/test_masks.py`.
- **Acceptance checks:**
  - Unmasked tokens decode to assistant turns only (sampled check).
  - At least ~50 of 60 tasks have one or more kept episodes.
  - Ledger total for this step is ≤ its budget slice.
  - The dataset is saved locally. The HF push is deferred to S8, or done earlier as private, with approval.
- **Risks:** mask bugs (P0), rate limits.

### S5: LoRA SFT on a Kaggle T4 (~3–5 h incl. runtime)
- **Outcome:** `train_kaggle.ipynb` runs TRL SFTTrainer on the pre-tokenized data. Settings: LoRA, fp16, gradient checkpointing, explicit max_length, batch 1 with gradient accumulation, a checkpoint at each epoch (2–3 total), and resume support.
- **Approach:** a smoke run on 10 examples first, then the full run.
- **Acceptance checks:**
  - Training loss decreases.
  - Each checkpoint loads and emits a parseable `<tool_call>` on a held-out train prompt.
  - No out-of-memory error at the chosen max_length.
- **Risks:** T4 out of memory on long sequences (fallback: lower max_length, drop the longest episodes, QLoRA), session timeouts, fp16 instability.

### S6: Dev evaluation, baselines, checkpoint choice and threshold (~3–5 h incl. runtime)
- **Outcome:**
  - vLLM with LoRA on the Kaggle T4, falling back to HF generate.
  - Run on the 14 dev tasks with k=3: B0, then each S2 checkpoint. Run the teacher on the Mac.
  - Freeze prompt and generation settings, and pick the best checkpoint on dev.
  - Run B0 and the teacher on the 40 test tasks.
  - Then **write `results/threshold.md`, dated (C9/AC10), before any S2 test run.**
- **Acceptance checks:**
  - Every record has every field.
  - The same user-sim settings are used for all models.
  - The threshold file's timestamp is earlier than the S2 test run.
  - Student throughput is measured.
- **Risks:** vLLM on Cohere2+LoRA on a T4 (fallback: HF), eval GPU time, user-sim variance.

### S7: Final test evaluation and analysis (~2–3 h)
- **Outcome:**
  - The chosen S2 checkpoint runs on the 40 test tasks with k=3.
  - Paired bootstrap CIs (AuditKIT if it accepts precomputed records, otherwise `metrics.py`).
  - Cost per success for the teacher (ledger) vs the student (GPU $/h ÷ measured successes per hour), with the license caveat (R2).
  - A failure-case sample.
- **Acceptance checks:**
  - AC4: ≥2 metrics, with reasoning.
  - AC6/AC7 numbers come from the records.
  - The threshold verdict is reported honestly.
- **Risk:** AuditKIT doesn't fit (fall back to the script).

### S8: Publish and submit (~2–3 h)
- **Outcome:**
  - Public, ungated HF model collection (the adapter, with a model card naming the base model, AC2).
  - Public HF dataset collection: split, raw traces, SFT set, eval records, threshold.
  - Public GitHub repo with a README covering reproducibility.
  - A write-up, an abstract under 500 words, and the submission form.
- **Acceptance checks:**
  - AC1, AC3, AC8 (a fresh re-run of the metrics from the published records reproduces the numbers) and AC9 (provenance and licenses listed).
- **Risk:** this is outward-facing. Each push or publication needs explicit approval at that moment.

## Optional (only if essential slices finish early)
- **O1, DAgger-lite (S3 in D1), ~5–8 h:**
  1. Run the S2 student on the 60 train tasks.
  2. Relabel turns up to the first invalid call or tool error. Failed episodes with neither have all their turns relabelled; this is a proposal, to confirm at the time.
  3. Retrain from the S2 adapter and evaluate on dev.
  4. Only if it wins on dev, run it on test, reporting both models transparently.
- **O2:** local Mac GGUF inference demo (P2, secondary to U2).
- **O3:** a small HF Space demo (repo link already satisfies O5).

## Cut lines if behind schedule
In order: drop O1–O3, then use k=2, then HF generate only, then skip AuditKIT, then K=4. A minimal submittable path is S1–S8 with B0 vs S2.

## Rough timeline (IST; solo)
- Nov 13: S0–S3.
- Nov 14: S4–S5 (training unattended overnight).
- Nov 15: S6–S7.
- Nov 16 morning: S8, submitting by about 12:00 IST to leave a 5-hour buffer before 17:29.

## Verification
- Each slice's acceptance checks (above) are run and reported with their actual outputs.
- End to end: the published records reproduce the reported metrics (AC8).
