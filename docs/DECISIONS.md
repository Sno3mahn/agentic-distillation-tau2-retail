# Decisions — Agentic Distillation

_Dated log. Each entry gives the decision, the reasons, the alternatives considered and its status. Requirement IDs (O/U/P/R/AC) refer to REQUIREMENTS.md._

## D1 — Solution direction (2026-11-13, user-decided in stage 4)
**Decision:**
- **Core (S2):** behaviour cloning. DeepSeek generates trajectories on the training tasks, only successful ones are kept (success-filtered, using the DB-state check), and Tiny Aya En-Thinker (3.35B, 32K context) is LoRA-SFT'd on them.
- **Next layer, if time permits (S3):** one DAgger-lite round.
  1. Run the S2 student on the training tasks.
  2. Have DeepSeek label the correct next action at the points where the student fails.
  3. Add those corrections to the data and run one more SFT round on the same model and adapter. No architecture change.
- **Fallback, if the organizers don't permit the thinker models (R8):** Tiny Aya Global (8K) with context compression, after measuring episode token lengths first.

**Reasons:**
- It's the simplest credible "after" result that fits ~2.5–3.5 days and 15–35 solo hours (U7).
- It matches AgentTune's own definition of "distill": behaviour cloning, i.e. SFT on the teacher's trajectories.
- The 32K student removes the 8K-context P0 (R1).
- S2 is complete on its own, so S3 can be dropped without breaking the submission.

**Alternatives considered:**
| Option | Why not chosen |
|---|---|
| B0: prompted base student only | Kept as the required "before" baseline (O7); not enough as a submission on its own |
| S1 with Tiny Aya Global | Retail episodes are estimated to exceed 8K (R1, unmeasured). Kept as the fallback |
| S4: BC + RL (GRPO via AgentTune) | Infeasible: rollouts on a T4 are slow, AgentTune integration is unknown work, and the deadline is close |
| S5: Aya Expanse 8B with QLoRA | Slow on a T4 with session limits; same 8K problem; weaker cost story |
| Custom mock API environment | Hours to build; less credible to judges than an established benchmark |
| BFCL multi-turn | Train split and license not verified this session; contamination risk |

**Status:** decided, but it depends on the organizers' answer on R8.

**Evidence that would reopen it:**
- the organizers' answer on the thinker models
- measured episode token lengths
- DeepSeek's success rate on the training tasks (too low leaves too little data)
- how strong B0 is (if strong, S3's focus on recovery matters more)
- schedule slippage (drops S3)

## D2 — Environment (2026-11-13, user-decided in stage 4)
**Decision:** τ²-bench **retail** domain, using its official split: 74 train / 40 test tasks, 17 tools.

**Reasons:**
- MIT license.
- Success is checked against the final database state, which can be done programmatically (P2) using the substring COMMUNICATE scorer or a DB-only basis (R5).
- An established benchmark, so credible to judges.
- The user simulator can be any LiteLLM model, so DeepSeek works.

**Alternatives considered:** τ²-airline (only 30 train tasks; kept as a possible supplement if data is short), a custom environment, BFCL (see D1).

**Status:** decided.

## D3 — Design decisions (2026-11-13, user-decided in stage 5)
| ID | Decision | Alternatives considered | Reason |
|---|---|---|---|
| D3.0 | R8 resolved: the user confirmed Tiny Aya with 32K context and thinking mode is allowed. **The student is Tiny Aya En-Thinker; the Global fallback is retired** | — | Organizer confirmation, relayed by the user |
| D3.1 | Thinking policy: **short visible thoughts**. Teacher in non-thinking mode; student in `/no_think` | actions only; distilled DeepSeek thinking | Some planning signal at low token, latency and cost (U2) |
| D3.2 | User simulator: **DeepSeek Flash for every run** | OpenRouter free models (first choice, reversed); DeepSeek Pro | OpenRouter `:free` is capped at 50 requests/day (1,000 with ≥$10 credit) [V], far below the ~9–12k requests needed |
| D3.3 | Dev split: **14 dev / 60 train**, from the 74 official train tasks; 40 test | 10/64; 20/54 | Enough to pick checkpoints and freeze prompts without touching test |
| D3.4 | **K = 6** teacher episodes per train task; **k = 3** eval trials per task | K=8/k=2; K=4/k=4; decide after pilot | Near-full task coverage after filtering; pass^3 and CIs; fits the $5 budget [A]. Re-checked at the S3 pilot gate |
| D3.5 | Tool protocol: **JSON in tags** (`<tool_call>`, `<tool_result>`) | Python call syntax; Command-R action format | Maps 1:1 to DeepSeek's JSON calls; strict, simple parsing |
| D3.6 | Invalid call: **one error feedback, then end the episode** | end immediately; N retries | Allows a single recovery; every error is counted |
| D3.7 | DAgger-lite: **relabel up to the first error, in failed episodes only** | every turn; all episodes | Targets the failure points at the lowest cost |
| D3.8 | Inference: **vLLM + LoRA, with HF `generate` as fallback** | HF only; HF Inference Endpoint | Throughput for eval time and a credible student cost figure (R3) |
| D3.9 | Lexsi repos: **AuditKIT for statistics only**; AgentTune off the critical path | try both, time-boxed; neither | U8 "if it fits", given the time |
| D3.10 | Run location: **API-only jobs (teacher generation and eval) on the Mac**; all Aya model work on Kaggle/Colab T4 | everything on Kaggle | The Mac's hardware isn't involved; faster iteration |

**Status:** decided.
**Open:** for failed DAgger-lite episodes that have no invalid call or tool error, relabelling all their turns is a proposal, to be confirmed at slice O1.

## D4 — Reward definition (2026-11-13, user-decided in S1)
**Decision:** reward = DB check × substring match on `communicate_info`, computed from τ²'s own `EnvironmentEvaluator` and `CommunicateEvaluator(scorer=substring)`. The NL_ASSERTION judge is not run. DB-only is also logged.

**Reasons:**
- At the pinned commit (`c06b07d`), 112 of 114 retail tasks score as DB × NL_ASSERTION; COMMUNICATE is in no task's basis.
- NL_ASSERTION is an OpenAI `gpt-5.4-mini` judge on 40 tasks: paid, outside the $5 budget, and against P2.
- 33 of those 40 tasks also have `communicate_info`, so the substring check covers most of what they test.

**Alternatives:** A, DB only (ignores what the agent says on ~35% of tasks); C, DeepSeek as NL judge (needs patching a hardcoded constant, and is an LLM judge); D, the official OpenAI judge (exact leaderboard score, but paid).

**Consequence:** reported numbers are not the official leaderboard score; the write-up must say so. **Status:** decided.

## D5 — Multi-call turns (2026-11-14, from S2a evidence)
**Decision:** an assistant turn is optional short text followed by zero or more `<tool_call>` blocks; results return in one user turn as `<tool_result name=…>` blocks, in call order. This refines D3.5.
**Reason:** in the 21 saved teacher episodes, 33 of 246 assistant turns carry 2–6 parallel calls, and 57 carry text plus a call. A one-call protocol would have dropped or split them.
**Status:** decided.

## D6 — Empty thinking block on every assistant turn (2026-11-14, user-confirmed)
**Decision:** the formatter renders `<|START_THINKING|><|END_THINKING|>` (the `/no_think` form) at the start of every assistant turn, in history as well as in the generation prompt. System and user turns follow Cohere's template exactly (tested).
**Reason:** the native template omits the thinking block for past assistant turns but includes it in the generation prompt, so the prefix property fails (checked: `False`). One-pass-per-episode SFT needs each turn to look as it did when generated.
**Alternative considered:** native history with one training example per assistant turn. Rejected: roughly 10× the sequences of 5–10K tokens on a T4.
**Cost:** history is non-native for both B0 and S2 (same for both, so the comparison stays fair). The effect on B0 is unmeasured. **Status:** decided.

## D7 — Stop at the first tool result (2026-11-14, user-approved after the S2b B0 run)
**Decision:** at inference, generation stops when the model starts a `<tool_result`; the text is cut there and the response block closed. Applied identically to B0 and S2.
**Evidence:** B0's first Kaggle run (task 75, 2026-11-14) wrote valid `<tool_call>` JSON but then hallucinated its own `<tool_result>` blocks and further calls with placeholder arguments (`your_user_id`); the strict parser rejected both attempts and the episode aborted on turn 1 (9.8K tokens in, 563 out, 45 s).
**Reason:** otherwise B0's score measures a formatting failure, not capability. The fine-tuned model learns to stop after its calls, so S2 is unaffected.
**Kept as evidence:** the pre-D7 run is cited in the write-up as B0 hallucinating tool results. **Status:** decided.

## D8 — Teacher variant and gate values (2026-11-14, user-decided at the S3 gate)
**Decision:** training data comes from the **rationale** variant (DeepSeek-Flash, non-thinking, system prompt asks for one short sentence before every tool call). **K = 6** episodes per train task, **k = 3** eval trials per task.
**Evidence (EXPERIMENT_LOG E1):** 10 tasks × 2 trials × 3 variants. Pass rates 85% / 95% / 100% (base / rationale / thinking), with paired CIs touching 0, so success differences are not established. Cost per episode $0.0020 / $0.0020 / $0.0031.
**Reasoning:** data is success-filtered, so teacher pass rate only sets yield (high for all variants), not quality. Rationale puts visible rationales on ~93% of tool-call turns (D3.1's short visible thoughts), at the lowest cost.
**Alternatives:** base (sparse visible text), thinking (hidden reasoning is not shown to the student; 1.55× cost), thinking + rationale (untested).
**Would reopen if:** the student underperforms, the kept data is too skewed to easy tasks, or the S2 comparison suggests richer thoughts. **Status:** decided.

## Open, for stage 5 (design). Resolved by D3 on 2026-11-13
- Thinking policy: no-think, short thoughts, or distilled DeepSeek thinking.
- The text format for tool calls and tool results, and the loss masks. No Aya template has a tool role (R4).
- Which model plays the simulated user, and its settings.
- Trials per task, for both data generation and evaluation.
- A dev split carved out of the 74 train tasks (AC5).
- Where AgentTune and AuditKIT fit (U8).

## D9 — Checkpoint for the test run, and the pre-registered bar (2026-11-15, user-approved)
**Decision:** test the **epoch3** adapter; freeze it by sha256 in `results/threshold.md` before any student test episode. The bar: student test pass^1 beats B0, with the paired 95% task-resampled bootstrap CI lower bound above 0.
**Evidence:** E4 (epoch3 42% vs epoch2 25% on 12 dev tasks, no aborts). The difference is within noise; epoch3 was also the lowest training loss.
**Alternatives:** epoch2; choosing after more dev trials (no GPU time). **Status:** decided, not to be changed after seeing test results.

## D10 — Crashes count as failures (2026-11-15)
**Decision:** an episode that crashes (e.g. CUDA OOM) is recorded with `terminated_by: "crash"`, reward 0, and stays in the denominator. Crash rate is reported per model, plus a sensitivity analysis excluding crashed episodes.
**Reasoning:** dropping crashes would favour the model that runs longer episodes (the trained student), because B0 aborts early and rarely reaches long contexts. **Status:** decided.

## D11 — Test set run across two environments (2026-11-15)
**Decision:** the "odd" test half ran on Kaggle (T4 x2); the "even" half runs on Colab (T4), same code commit and adapter hash, greedy decoding. B0 runs all 40 tasks on Kaggle.
**Reasoning:** Kaggle attempt 1 lost GPU 0 to a bug; rerunning all 40 would cost ~5 GPU hours before the deadline. **Cost:** torch/transformers versions may differ, which can change greedy outputs slightly. Recorded in `results/env_colab.json` and noted as a limitation. **Status:** decided.

## D12 — Continued training is a separate, post-hoc experiment (2026-11-15, user-approved)
**Decision:** S8's extra epochs (E6) are evaluated on dev only and reported as an exploratory result. They do not replace epoch3 as the pre-registered test result and are not run on test unless the user decides otherwise, in which case both results are reported. **Status:** decided.
