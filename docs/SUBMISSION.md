# Submission sheet (form fields)

## Track / problem statement
Track 3, problem statement 3: agentic distillation.

## Links
- GitHub (code, write-up = README): https://github.com/Sno3mahn/agentic-distillation-tau2-retail
- HF model collection: https://huggingface.co/collections/ritabrata/agentic-distillation-retail-models-6afa0df9dbeb2613fc80c373
- HF dataset collection: https://huggingface.co/collections/ritabrata/agentic-distillation-retail-datasets-6afa0dfb6a5b1af4d9af1ef8
- Model: https://huggingface.co/ritabrata/tiny-aya-en-thinker-tau2-retail-lora (tag `v1-epoch3`)
- Dataset: https://huggingface.co/datasets/ritabrata/agentic-distillation-tau2-retail-data (tag `v1`)

## Repos, models, data used
- Core model (Cohere Labs): `CohereLabs/tiny-aya-en-thinker` (3.35B, CC-BY-NC-4.0), LoRA adapter trained here.
- Supplementary model: DeepSeek V4.1 Flash (API) as teacher and as user simulator.
- Environment / benchmark: τ²-bench retail, `sierra-research/tau2-bench` at commit `c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4` (MIT).
- Libraries: transformers, peft, torch, LiteLLM (through τ²-bench).
- Lexsi repos: AgentTune and AuditKIT were evaluated and not used. AgentTune's distill stage documents open-weight
  teachers only (the teacher here is an API); AuditKIT's documented evaluators do not cover multi-turn tool-call
  episodes. The definition of "distill" used here is AgentTune's: behaviour cloning on a teacher's captured
  trajectories. Statistics (pass^k, task-resampled bootstrap) are in `src/agentic_distill/metrics.py`.

## Abstract (v1: epoch3 vs baseline, the pre-registered result)

**Distilling a tool-calling agent into a 3.35B model: what transfers, what does not, and what it costs**

We distil a multi-turn, tool-calling customer-service agent from an API teacher (DeepSeek V4.1 Flash) into Tiny Aya
En-Thinker (3.35B, LoRA) and measure the result in a stateful environment rather than on static call matching. The
environment is τ²-bench retail: 17 tools, a written policy, an LLM-simulated user, and a reward that checks the final
database state and the facts told to the user (substring scorer, no LLM judge). Tiny Aya has no tool-calling chat
template, so we define a JSON-in-tags call format with one shared formatter for training and inference.

Method: behaviour cloning. The teacher ran 60 training tasks; only fully successful episodes were kept (370
trajectories, 59 tasks). The student was trained for 3 epochs with loss on assistant turns only, on a free T4. The
official test split (40 tasks) was never used for training, checkpoint choice or prompt tuning; 14 dev tasks held out
of train served those purposes. The success bar and the frozen adapter hash were written down before any student
test episode.

Results on 40 held-out tasks × 3 trials:

| | pass^1 | pass^3 | valid tool calls | loop/abort |
|---|---|---|---|---|
| base model, prompted | 2.5% | 2.5% | 7.3% | 30.8% |
| distilled student | 40.0% | 10.0% | 88.7% | 10.0% |
| teacher | 89.2% | 72.5% | 97.7% | 0% |

The paired gain over the base model is +37.5 points of pass^1 (95% CI [+25.8, +48.3], resampling tasks), so the
pre-registered bar is met. The untrained model almost never produces a usable call; the student completes two in five
unseen tasks end to end.

Three findings are negative, and we report them as such. (1) Reliability does not transfer as well as capability:
the student solves a task in all three trials only 10% of the time, against 72.5% for the teacher. (2) More epochs of
the same data did not help on dev (invalid calls fell, pass rate did not), which points at compounding errors rather
than under-fitting; a one-round DAgger experiment with a pre-registered plan addresses this. (3) The cost case is not
made yet: served naively (HF `generate`, no prefix caching) on a rented T4, the student costs about $0.12 per
successful task against $0.0023 for the teacher API, and its CC-BY-NC licence excludes commercial use. The student is
the better choice only where data must stay on local hardware or where an efficient serving stack (cached prefixes,
batching) is in place; we state the measurement and the assumption instead of a favourable estimate.

Everything needed to check the claims is public: code, the task split, raw teacher traces, the training set, every
evaluation episode and trace, the pre-registration files, and a script that rebuilds the result table from the
published records.

## Abstract (v2: use only if resubmitting with the DAgger variants)

**Distilling a tool-calling agent into a 3.35B model: what transfers, what does not, and what it costs**

We distil a multi-turn, tool-calling customer-service agent from an API teacher (DeepSeek V4.1 Flash) into Tiny Aya
En-Thinker (3.35B, LoRA) and measure it in a stateful environment, τ²-bench retail: 17 tools, a written policy, an
LLM-simulated user, and a reward that checks the final database state and the facts told to the user (substring scorer,
no LLM judge). Tiny Aya has no tool-calling template, so we define a JSON-in-tags call format with one formatter shared
by training and inference.

Method: behaviour cloning on 370 teacher trajectories that fully succeeded (59 tasks), 3 epochs, loss on assistant turns
only, on a free T4. The 40-task official test split was never used for training, checkpoint choice or prompt tuning.
The success bar and the frozen adapter hash were written down before any student test episode.

Results on 40 held-out tasks × 3 trials:

| | pass^1 | pass^3 | valid tool calls |
|---|---|---|---|
| base model, prompted | 2.5% | 2.5% | 7.3% |
| distilled student (headline) | 40.0% | 10.0% | 88.7% |
| + DAgger round 1 | 36.7% | 17.5% | 91.5% |
| + DAgger round 2 | 46.7% | 30.0% | 90.5% |
| teacher | 89.2% | 72.5% | 97.7% |

The paired gain of the student over the base model is +37.5 points of pass^1 (95% CI [+25.8, +48.3], resampling tasks);
the pre-registered bar is met. Reliability is the weak point: the student succeeds on all three trials of a task only
10% of the time, against 72.5% for the teacher.

We then tried DAgger, where the teacher labels states the student itself reached, with plans fixed in advance. Round 1
(one unfiltered teacher turn per state) did not help (pass^1 −3.3 points, CI [−13.3, +7.5]). Round 2 keeps a label only
if a teacher rollout from that state succeeds. It gives +6.7 points of pass^1 (CI [−5.0, +18.3]), which does not meet
the pre-registered rule, and +20.0 points of pass^3 (CI [+5.0, +37.5]). The pass^3 gain is a secondary metric chosen
after the fact, among several comparisons, so we report it as suggestive. Two more plain epochs did not help on dev.

The cost case is not made: served naively (HF `generate`, no prefix caching) on a rented T4, the student costs about
$0.12 per successful task against $0.0023 for the teacher API, and its CC-BY-NC licence excludes commercial use.

All code, splits, teacher traces, training sets, every evaluation episode and the pre-registration files are public,
with a script that rebuilds the tables from the published records.
