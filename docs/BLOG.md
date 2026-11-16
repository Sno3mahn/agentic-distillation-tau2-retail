# Distilling a tool-calling agent into a 3.35B model: what transferred, what didn't, and what it cost

*Lexsi AI Safety & Alignment Hackathon, Track 3, problem statement 3 (agentic distillation). Code: [GitHub](https://github.com/Sno3mahn/agentic-distillation-tau2-retail) · Model: [LoRA adapter](https://huggingface.co/ritabrata/tiny-aya-en-thinker-tau2-retail-lora) · Data: [traces, splits, every eval episode](https://huggingface.co/datasets/ritabrata/agentic-distillation-tau2-retail-data)*

## The question

Can a small open model learn to act as a multi-turn, tool-calling agent by copying a much stronger one, and is that worth it?

We used **DeepSeek V4.1 Flash** as the teacher and **Tiny Aya En-Thinker** (3.35B, Cohere Labs) as the student, trained with LoRA. The task is **τ²-bench retail**: a customer-service agent with 17 tools (look up orders, cancel, exchange, return, modify), a 6.7 KB policy it must follow, and an LLM-simulated customer. Success is checked mechanically: the final database state must match the gold state, and the agent must have told the user the required facts (a substring check, no LLM judge). That makes "did it work" a measurement, not an opinion, and it makes the failure modes visible. A model can sound fluent and still cancel the wrong order.

## Setup, and the rules we set before looking

- **Split.** τ²'s official 40 test tasks were locked in code until a pre-registration file existed. The 74 train tasks were split by a seeded shuffle into 60 train and 14 dev. Test tasks were never used for training, checkpoint choice, or prompt tuning.
- **Pre-registration.** Before any student test episode, we wrote down the success bar (the lower bound of a 95% CI on the student's pass^1 gain over the untrained base model must exceed 0) and froze the adapter by its sha256. The file is in the repo with its timestamp.
- **Metrics.** pass^1 (one attempt succeeds) and pass^3 (all three attempts at a task succeed; the τ-bench reliability measure), plus valid-tool-call rate and loop/abort rate. Confidence intervals resample **tasks**, not episodes, because trials of one task aren't independent. Crashed episodes count as failures.
- **Three models, same harness:** the untrained base ("B0"), the trained student, and the teacher itself.

## What we built

**A call format.** Tiny Aya's chat template has no tool role, and none of the Aya templates have the `{% generation %}` markers that "train only on the assistant" tooling expects. So we defined our own: tool calls as JSON inside `<tool_call>` tags, tool results in `<tool_result>` blocks, one formatter shared by training and inference, with round-trip tests against real teacher traces.

**Teacher data.** 60 train tasks × 6 teacher episodes, non-thinking mode with a one-sentence rationale before each action, plus a thinking-mode rerun on 17 hard tasks. We kept only episodes that **fully succeeded** (reward = 1), at most 8 per task: **370 trajectories covering 59 of 60 tasks**. The data cost about $1.32 in API calls. Behaviour cloning on successes is the simplest distillation recipe, and it is what Lexsi's own AgentTune describes as "distill".

**Training.** LoRA (r=16) on all attention and MLP projections, 3 epochs, fp16 on a free T4, loss on assistant turns only. Two engineering details mattered:

- The vocabulary is 262K tokens, so the full logits tensor for a long episode doesn't fit on a T4. We apply the output head **only at supervised positions, in chunks**, and test that the loss and gradients equal the stock implementation.
- A single episode can run to ~13K tokens. We exclude over-length episodes (and count them) rather than truncate, since silent truncation would train on cut-off conversations.

## Result

40 held-out tasks × 3 trials each (120 episodes per model):

| | pass^1 | pass^3 | valid tool calls | loop/abort |
|---|---|---|---|---|
| Base model, prompted | 2.5% [0, 7.5] | 2.5% | 7.3% | 30.8% |
| **Distilled student** | **40.0% [30, 50]** | **10.0% [2.5, 20]** | **88.7%** | **10.0%** |
| Teacher (DeepSeek) | 89.2% [82.5, 95] | 72.5% [57.5, 85] | 97.7% | 0% |

The paired gain of the student over the base model is **+37.5 points of pass^1, 95% CI [+25.8, +48.3]**. The pre-registered bar is met, and excluding the 3 crashed episodes gives +38.7 [+27.1, +50.0].

What the untrained model does is instructive: in 84 of its 120 episodes it ends by an error, mostly malformed calls, because it was never trained to emit this format. Training mostly teaches *how to operate the tools*. It then gets roughly two tasks in five right end to end.

The gap that remains is **reliability**. The student solves a task in all three trials only 10% of the time; the teacher does 72.5%. And 60 of the student's 72 failed episodes finished normally. They weren't format failures, they were wrong decisions: a wrong item, a wrong argument, a skipped policy step. More epochs of the same data cut invalid calls on dev (19 → 11 → 8) without moving the pass rate (3/8 vs 4/8), which says the same thing.

## Trying to fix reliability: DAgger

Behaviour cloning only sees states the *teacher* reached. Once the student makes a mistake it is in a state it never saw in training, and errors compound. DAgger addresses this: run the student, find where it goes wrong, and have the teacher say what it should have done *from the student's own state*, then train on those labels.

**Round 1.** From the student's failed train episodes we picked 54 states (its first error turn and earlier write calls), asked the teacher for one turn at each, and trained on 50 labels plus a replay sample of the original data. Test result: pass^1 36.7%, a paired **−3.3 points [−13.3, +7.5]** against the student. Valid-call rate rose a little (88.7 → 91.5%), but pass rate didn't. By the pre-registered rule: no improvement.

The obvious weakness: a single teacher turn has no reward, so we couldn't tell whether the label leads anywhere good.

**Round 2, success-filtered.** Roll the teacher forward from each student state to the end of the episode, four times, score the full trajectory, and keep a label only if a successful rollout starts with it. 110 of 204 rollouts succeeded; 31 of 51 states had a usable label and 20 were dropped.

A scoring trap cost us a restart. τ²'s evaluator builds its gold database by replaying the same message history we resume from. If the student's prefix already contained a *wrong* write, the gold state contained it too, and a rollout could pass without fixing anything. We caught it from the warnings "Non-pending order cannot be cancelled", discarded the first 60 rollouts, and re-scored every rollout against the original task from scratch.

Test result (same 40 tasks × 3 trials, same environment as round 1):

| | pass^1 | pass^3 |
|---|---|---|
| Student | 40.0% [30, 50] | 10.0% [2.5, 20] |
| DAgger round 1 | 36.7% [25.8, 48.3] | 17.5% [5, 30] |
| **DAgger round 2** | **46.7% [33.3, 60]** | **30.0% [17.5, 45]** |

Paired against the student, round 2 is **+6.7 pts pass^1 [−5.0, +18.3]** and **+20.0 pts pass^3 [+5.0, +37.5]**.

**How we read it.** The pre-registered rule asked about pass^1, and it fails (the lower bound is −5.0). The pass^3 gain does exclude zero, but we chose that metric after seeing the results, among several comparisons across two variants, so we call it **suggestive, not confirmed**. The student's headline result stays the pre-registered one. What we'd claim: success-filtering the labels appears to help reliability more than raw accuracy, which fits the mechanism, and a fresh pre-registered run would be the way to find out.

## What didn't work, and what we'd distrust

- **Continued training.** Epochs 4 and 5 didn't help on dev.
- **Unfiltered DAgger labels** didn't help; filtered ones might.
- **Cross-machine noise.** The test set ran on Kaggle, Colab and Modal because of quota and a bug that lost half a run. Greedy decoding differs slightly across library versions: the same adapter scored 5/12 on the dev tasks on Kaggle and 1/14 on Modal. All DAgger variants ran only on Modal, while the student's test records are pooled across three machines, so that comparison carries a confound.
- **Crashes.** Three of 120 student test episodes hit GPU out-of-memory on very long contexts. We score them as failures and report the sensitivity.
- **The scorer.** The substring COMMUNICATE check can pass a reply that states a required number wrongly; we didn't audit this.
- **Scale.** 40 tasks give CIs of about ±10 points. Differences smaller than that, including every DAgger comparison, are within noise on pass^1.

## What it costs, honestly

The hackathon rewards a cost story, and ours is a negative one. The teacher API costs about **$0.0023 per successful task**. Our student, served with plain Hugging Face `generate` on a rented T4 (~$0.59/h), costs about **$0.12 per successful task**: every turn re-processes the whole conversation (median 132K input tokens per episode, around 5 minutes per episode), and it succeeds less often. That is 50× worse than the API, so on cost alone the student loses here. A serving stack with prefix caching and batching (vLLM and similar) would likely close much of the gap, but we did not run it, so we don't claim it. The student's licence (CC-BY-NC-4.0) also rules out commercial use without a Cohere licence. The case for the student is local deployment and data that can't leave the machine, not price.

## What we'd do next

1. Serve with prefix caching and re-measure cost per success.
2. More and better on-policy states: several DAgger rounds, and state selection based on the first *wrong decision*, since most failures are clean but wrong.
3. Replicate round 2 with more tasks and trials in a single environment, with pass^3 pre-registered.
4. Audit the COMMUNICATE scorer semantically.

## Reproduce

Everything is public: the split, the teacher traces, the SFT sets, every evaluation episode and trace, the pre-registration files, and the scripts. The result tables rebuild from the published records with `PYTHONPATH=src python scripts/pool_test.py` and `scripts/pool_dagger.py`. Experiment-by-experiment notes and decisions are in `docs/EXPERIMENT_LOG.md` and `docs/DECISIONS.md` in the GitHub repo.

*Models: Tiny Aya En-Thinker (CohereLabs, CC-BY-NC-4.0), DeepSeek V4.1 Flash (teacher and user simulator). Environment: τ²-bench retail (MIT).*
