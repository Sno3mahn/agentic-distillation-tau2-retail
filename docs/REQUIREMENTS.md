# Requirements — Agentic Distillation (Lexsi Hackathon, Track 3, PS 3)

_Stage 2 (requirements discovery), drafted 2026-11-11. Updated 2026-11-12 after stage 3 research._

**Labels:** **[O]** organizer-stated · **[U]** project decision · **[I]** inferred · **[P]** proposal that was approved (see section 4) · **[R]** constraint or risk found in research (see section 6)
**Evidence tags:** [V] verified from a primary source · [M] well known, cited from memory · [C] unverified community claim · [A] arithmetic, not a measurement

**Sources:**
- Organizer pages: verbatim copies of hackathon.lexsi.ai /submit, /eligibility, /faqs and /structure, copied by the author on 2026-11-12. /rules and /tracks are paraphrased from the home page (fetched 2026-11-11).
- agenttune.lexsi.ai: home, agentic-spine and FAQ pages. auditkit.lexsi.ai: home page.
- Anthropic Consumer Terms (effective 2025-10-08). DeepSeek Open Platform ToS (effective 2026-04-29).
- Stage 3 research sources are linked in `DOMAIN_GLOSSARY.md`.

## 1. Organizer-stated requirements [O]
- O1. Build during the hackathon. The build window is Nov 1–14; **submissions are due Nov 15, 23:59 AoE**. You can submit more than once; only the latest submission counts.
- O2. The core model must be a dense, open-weights Cohere Labs model: Tiny Aya (Base, Global, Fire, Water or Earth) or Aya Expanse 8B/32B. Other models may be used only as supplements, such as a teacher, validator or LLM judge. *Note (2026-11-13):* the newer Cohere Labs releases (tiny-aya-en-thinker, tiny-aya-l2-thinker, tiny-aya-base-32K) are not on this list. Whether they are permitted still needs organizer confirmation (R8).
- O3. Track 3 is exempt from the 2-repo minimum ("At least 2 repos from your track, used together (except Track 3)"). Using 3 or more is preferred. Track 3 tools: AgentTune ("post-training agents on tool-call accuracy and efficiency") and AuditKIT ("evaluating tool-call accuracy and reliability").
- O4. Open-source libraries and public datasets are allowed.
- O5. Everything we create goes on Hugging Face:
  - **Model collection:** all models and adapters.
  - **Dataset collection:** "all datasets, logs, traces, benchmarks, curated or synthetic data".
  - **Both collections and every individual repo in them must be public and ungated.**
  - The project must also be publicly deployed by the deadline, as a demo or a repo link.
- O6. Submission package:
  - registered team member's email
  - abstract under 500 words
  - HF model collection and HF dataset collection
  - write-up (HF article, or a GitHub README as fallback)
  - GitHub repo with code or notebooks, and reproducibility notes in the README
  - list of the repos, models, tracks and problem statements used
- O7. Before/after results on **more than one metric**, with reasoning for the metric, dataset and model choices, and why the resulting model is better for use.
- O8. Optional but preferred: an ROI comparison of a closed-source API vs the self-hosted fine-tuned small model.
- O9. Judging: judges score numerically on a fixed scale, and up to 20 teams are shortlisted by Nov 16. Shortlisted teams submit a video or slides by Nov 22, 23:59 AoE, then the final panel picks the winners. **Scoring criteria and weights are not published.**
- O10. Compute offered:
  - $20 of HF credits (Discord teams, first 800 to redeem). Credits expire 30 days after redemption, and redeeming may require adding a payment method.
  - "Google Colab and Kaggle also give generous free GPU quotas, which can be enough for the Tiny Aya models."
  - Finalists get $200 more.

## 2. Project decisions [U]
| ID | Date | Decision |
|---|---|---|
| U1 | 2026-11-11 | This is a real hackathon submission. The organizer rules and the deadline are hard constraints. |
| U2 | 2026-11-11 | Lower running cost is the main goal. "Runs locally" is secondary, pursued only if feasible. |
| U3 | 2026-11-11 | Domain: general multi-step tool/API calling, **English only**. Multilingual is future work. |
| U4 | 2026-11-11, amended 2026-11-13 | Compute: $20 HF credits plus Kaggle/Colab only. The Mac is not used for training, or for running any Aya model. *Amended 2026-11-13:* API-only jobs (teacher generation and teacher evaluation) run on the Mac (D3.10). |
| U5 | 2026-11-11 | Teacher: the **DeepSeek API, at most $5 in total** (trajectory generation plus teacher baseline). **Open datasets and existing high-quality teacher traces complement** the DeepSeek calls. |
| U6 | 2026-11-11 | No Anthropic model is used as a teacher; a consumer subscription is excluded (Consumer Terms §3 forbid automated access and training models on outputs). The Anthropic API route is not being pursued. |
| U7 | 2026-11-11 | Capacity: solo, 15–35 hours before the deadline. |
| U8 | 2026-11-11 | Lexsi repos (AgentTune, AuditKIT) are preferred if they fit; otherwise use standard libraries. |
| U9 | 2026-11-11 | The student's size class (Tiny Aya vs Aya Expanse 8B) and variant are still open; research evidence is now in. |
| U10 | 2026-11-11 | The success threshold is set after the baselines are measured. |

## 3. Inferences and research facts [I]
- I1. Aya Expanse 8B [V]: 8K context, CC-BY-NC-4.0, gated. Inference [A, unmeasured]: in fp16 its weights alone (~16 GB) fill a T4, so it needs QLoRA, which is slow. Aya Expanse 32B is impractical as a student on free-tier GPUs.
- I2. *Resolved* [V]: Tiny Aya Global has **3.35B parameters**, 8K input / 8K output, CC-BY-NC-4.0, and is gated. Its model card doesn't mention tool use.
  - *Added 2026-11-13* [V]: Tiny Aya En-Thinker and L2-Thinker (released 2026-10-30) and Base-32K (2026-09-30) are all 3.35B, **32K context** and CC-BY-NC.
  - The thinkers have dual-mode reasoning. `enable_thinking` appends `/think` or `/no_think` to the user turn, and the model writes its thinking between `<|START_THINKING|>…<|END_THINKING|>`.
  - En-Thinker reasons in English; L2-Thinker reasons in the prompt's language.
- I3. *Revised* [A]: with DeepSeek V4.1-Flash prices (off-peak: $0.15/M uncached input, $0.003/M cached input, $0.60/M output; peak is 2×) and caching on by default, an episode costs roughly $0.002–0.01. **$5 is therefore roughly 500–2,500 episodes.** This is arithmetic, not a measurement.
- I4. Public traces can be used only if their license permits training and redistribution on HF (O4, O5). This is checked per dataset; see R6.
- I5. Evaluation runs on the same limited GPUs, so the held-out set has to stay small, which makes results noisy (see AC4).

## 4. Approved proposals [P] (approved 2026-11-12)
- P1. Evaluate the base student, the trained student and the teacher on the same tasks with the same harness.
- P2. Verify task success programmatically (final state or call correctness), not with an LLM judge as the primary signal.
- P3. Candidate metrics: task success rate, valid tool-call rate, steps per task and loop/abort rate, and cost per successful task.
- P4. Fix the success threshold before evaluating the trained student on held-out tasks.
- P5. Scope: one student, one training method, one environment, one main training run, and at most one ablation.

## 5. Acceptance criteria (observable; no thresholds invented)
- AC1 [O1, O6]: The submission form is filed before Nov 15, 23:59 AoE, with every O6 item present.
- AC2 [O2]: The model card names a listed Cohere Aya model as the base of the published student.
- AC3 [O5]: The HF model and dataset collections, and every repo in them, open without logging in or requesting access. The dataset collection contains the training trajectories, the evaluation logs and traces, and the benchmark split.
- AC4 [O7]: The write-up reports base-student vs trained-student results on at least 2 named metrics, each with its reasoning.
- AC5 [CLAUDE.md]: The evaluation tasks are disjoint from the training and tuning tasks. The split method is documented, and the overlap is shown to be zero.
- AC6 [O7, U2]: The write-up explains why the trained student is better to use, with cost as the main argument.
- AC7 [O8, U2]: The write-up reports the teacher's API cost per task alongside the student's self-hosted cost per task. It states the cost assumptions, including the serving setup and hardware price (R3), and the license caveat (R2).
- AC8 [O6]: Following the README's reproducibility notes, someone else can re-run the evaluation on the published student and get the reported numbers.
- AC9 [U5, I4]: Every trajectory source (DeepSeek-generated or public) is listed with its license or terms, and each one permits training and redistribution.
- AC10 [U10, P4]: The success threshold is written down, with its date, before the trained student's held-out evaluation.

## 6. Research-derived constraints and risks [R] (2026-11-12)
| ID | Priority | Constraint / risk | Evidence |
|---|---|---|---|
| R1 | P0 | Both Aya students have an **8K context**. The schemas, policy, history and next turn must all fit. TRL's SFT `max_length` defaults to **1024**, and its `keep_start` truncation silently drops later turns. Episode token lengths must be measured before committing to an environment. *2026-11-13:* the 32K En-Thinker student (D1) avoids the 8K limit, but it still applies to the Global fallback. The `max_length` trap applies to every student. | Model cards [V]; TRL v1.17.1 docs [V] |
| R2 | P1 | The student inherits **CC-BY-NC-4.0**. Publishing it for the hackathon is fine, but a real API-replacement deployment would need a Cohere commercial license. The cost story must state this caveat. | Model card; CC deed [V] |
| R3 | P1 (to U2) | A cheap, cached DeepSeek teacher may **cost less per task** than a lightly used self-hosted student. The cost comparison point and the serving and utilization assumptions must be explicit. | DeepSeek pricing and caching [V]; HF prices [V]; arithmetic [A] |
| R4 | Resolved 2026-11-13 | **No Aya chat template has a tool role, a tool-schema slot or `{% generation %}` tags.** This covers Global, both thinkers and Expanse 8B. Global's template enforces user/assistant alternation and drops the content of other roles. So we need our own text format for tool calls and results and our own loss masks: a known task, no longer a risk. No model card mentions tools. | Templates read with HF token [V] |
| R5 | P1 | τ²-bench's default reward is DB × COMMUNICATE, and COMMUNICATE uses an LLM judge by default. Meeting P2 needs the substring scorer or a DB-only reward basis. **Update 2026-11-13 (see D4):** at the pinned commit, 112 of 114 retail tasks have `reward_basis = [DB, NL_ASSERTION]` and none has COMMUNICATE. NL_ASSERTION is an OpenAI LLM judge (`gpt-5.4-mini`) that runs on 40 tasks. We therefore compute DB × substring-COMMUNICATE ourselves, so our numbers are **not** the official leaderboard score. | τ²-bench evaluation docs [V]; pinned-commit task data [V] |
| R6 | P1 | APIGen-MT-5k likely fails AC9: CC-BY-NC, research only, an OpenAI non-compete clause, and possible overlap with τ-bench's test tasks [unverified]. xLAM-60k is CC-BY-4.0, but its "research only" note is ambiguous. | Dataset cards [V] |
| R8 | Resolved 2026-11-13 | The chosen student (Tiny Aya En-Thinker, D1) was not on the organizers' permitted list (O2). **It was confirmed that Tiny Aya with 32K context and thinking mode is allowed** (D3.0). | User confirmation |
| R9 | P1 (noted 2026-11-13) | OpenRouter `:free` models are capped at 20 requests/min and 50/day (1,000/day after buying ≥$10 of credit). That's unusable as the user simulator at our volume, so DeepSeek Flash is used instead (D3.2). | OpenRouter limits docs [V] |
| R7 | P0 if missed | T4/P100 GPUs have no bf16, and TRL defaults to bf16, so fp16 must be set explicitly. FlashAttention-2 is unavailable on T4, so `padding_free` can't be used. | TRL docs [V]; hardware [M] |

## 7. Open items
_2026-11-13:_ the stage-4 choices (direction and environment) are recorded in `DECISIONS.md` (D1, D2), together with the design-stage open items. The rows below are kept for history. Distill scope, student, environment and AgentTune/AuditKIT are now tracked there.

| Item | Resolve in | Notes |
|---|---|---|
| Scope of "distill" (narrow vs broad) | Stage 4 | Behaviour cloning (SFT on the teacher's trajectories) is feasible with an API teacher, and it is how AgentTune defines "distill". On-policy correction addresses compounding errors but costs live teacher calls. Logit KD is impractical (different tokenizers, no logprobs). |
| Student size and variant (U9) | Stage 4 | Evidence: Tiny Aya (3.35B) fits LoRA on a T4; 8B needs QLoRA. Both are 8K context and CC-BY-NC. Tool support is unverified (R4). |
| Tool environment / dataset and held-out source | Stage 4 | Must satisfy P2, AC5, AC9 and R1. |
| Reasoning-trace policy (none / short / long) | Stage 4 | DeepSeek thinking mode is on by default; it trades off against the 8K window (R1). |
| Cost comparison point and serving setup | Stage 4 | R2, R3, AC7. |
| AgentTune fit | Stage 4 | SFT distill stage with `EventLog` trajectories and a `teacher_engine`. API teachers are undocumented; requires Python 3.12+. |
| AuditKIT fit | Stage 4 | Its docs show no agent or tool-call evaluation, which conflicts with the track page's description. It does report cost automatically and run bootstrap significance tests. |
| Success threshold | After baselines | U10 / P4 / AC10. |
| Organizer view on repo usage for Track 3 | Optional: Discord | O3. |
| Local Mac inference demo via GGUF | Later (P2 priority) | Unverified for Tiny Aya. Secondary under U2. |
