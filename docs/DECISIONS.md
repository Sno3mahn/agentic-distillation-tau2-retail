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

## Open, for stage 5 (design). Resolved by D3 on 2026-11-13
- Thinking policy: no-think, short thoughts, or distilled DeepSeek thinking.
- The text format for tool calls and tool results, and the loss masks. No Aya template has a tool role (R4).
- Which model plays the simulated user, and its settings.
- Trials per task, for both data generation and evaluation.
- A dev split carved out of the 74 train tasks (AC5).
- Where AgentTune and AuditKIT fit (U8).
