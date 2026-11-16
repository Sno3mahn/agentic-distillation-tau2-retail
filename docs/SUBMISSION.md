# Submission sheet (form fields)

## Track / problem statement
Track 3, problem statement 3: agentic distillation.

## Links
- GitHub (code, write-up = README): https://github.com/Sno3mahn/agentic-distillation-tau2-retail
- HF model collection: https://huggingface.co/collections/ritabrata/agentic-distillation-retail-models-6afa0df9dbeb2613fc80c373
- HF dataset collection: https://huggingface.co/collections/ritabrata/agentic-distillation-retail-datasets-6afa0dfb6a5b1af4d9af1ef8
- Model: https://huggingface.co/ritabrata/tiny-aya-en-thinker-tau2-retail-lora (tags `v1-epoch3` = headline, `v2-dagger` = + DAgger variants)
- Dataset: https://huggingface.co/datasets/ritabrata/agentic-distillation-tau2-retail-data (tags `v1`, `v2-dagger`)

## Repos, models, data used
- Core model (Cohere Labs): `CohereLabs/tiny-aya-en-thinker` (3.35B, CC-BY-NC-4.0), LoRA adapter trained here.
- Supplementary model: DeepSeek V4.1 Flash (API) as teacher and as user simulator.
- Environment / benchmark: τ²-bench retail, `sierra-research/tau2-bench` at commit `c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4` (MIT).
- Libraries: transformers, peft, torch, LiteLLM (through τ²-bench).
- Lexsi repos: AgentTune and AuditKIT were evaluated and not used. AgentTune's distill stage documents open-weight
  teachers only (the teacher here is an API); AuditKIT's documented evaluators do not cover multi-turn tool-call
  episodes. The definition of "distill" used here is AgentTune's: behaviour cloning on a teacher's captured
  trajectories. Statistics (pass^k, task-resampled bootstrap) are in `src/agentic_distill/metrics.py`.

## Abstract (plain text, no markdown; paste as is)

Distilling a tool-calling agent into a 3.35B model: what transfers, what does not, and what it costs

We distil a multi-turn, tool-calling customer-service agent from an API teacher (DeepSeek V4.1 Flash) into Tiny Aya En-Thinker (3.35B, LoRA) and measure it in a stateful environment, tau2-bench retail: 17 tools, a written policy, an LLM-simulated user, and a reward that checks the final database state and the facts told to the user (substring scorer, no LLM judge). Tiny Aya has no tool-calling template, so we define a JSON-in-tags call format with one formatter shared by training and inference.

Method: behaviour cloning on 370 teacher trajectories that fully succeeded (59 tasks), 3 epochs, loss on assistant turns only, on a free T4. The 40-task official test split was never used for training, checkpoint choice or prompt tuning. The success bar and the frozen adapter hash were written down before any student test episode.

Results on 40 held-out tasks x 3 trials (pass^1 = one attempt succeeds; pass^3 = all three attempts at a task succeed):
- Base model, prompted: pass^1 2.5%, pass^3 2.5%, valid tool calls 7.3%
- Distilled student (headline): pass^1 40.0%, pass^3 10.0%, valid tool calls 88.7%
- Student + DAgger round 1: pass^1 36.7%, pass^3 17.5%, valid tool calls 91.5%
- Student + DAgger round 2: pass^1 46.7%, pass^3 30.0%, valid tool calls 90.5%
- Teacher (DeepSeek): pass^1 89.2%, pass^3 72.5%, valid tool calls 97.7%

The paired gain of the student over the base model is +37.5 points of pass^1 (95% CI +25.8 to +48.3, resampling tasks); the pre-registered bar is met. Reliability is the weak point: the student succeeds on all three trials of a task only 10% of the time, against 72.5% for the teacher.

We then tried DAgger, where the teacher labels states the student itself reached, with plans fixed in advance. Round 1 (one unfiltered teacher turn per state) did not help (pass^1 -3.3 points, CI -13.3 to +7.5). Round 2 keeps a label only if a teacher rollout from that state succeeds. It gives +6.7 points of pass^1 (CI -5.0 to +18.3), which does not meet the pre-registered rule, and +20.0 points of pass^3 (CI +5.0 to +37.5). The pass^3 gain is a secondary metric chosen after the fact, among several comparisons, so we report it as suggestive, not confirmed. Two more plain epochs did not help on dev.

The cost case is not made: served naively (HF generate, no prefix caching) on a rented T4, the student costs about $0.12 per successful task against $0.0023 for the teacher API, and its CC-BY-NC licence excludes commercial use.

All code, splits, teacher traces, training sets, every evaluation episode and the pre-registration files are public, with a script that rebuilds the tables from the published records.
