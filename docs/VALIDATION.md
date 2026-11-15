# Validation

Held-out evaluation of the distilled student (stage 10). Numbers are reproduced by `scripts/pool_test.py` from the
records in `results/test_pooled_*.jsonl`.

## Method
- **Tasks:** the 40 official τ²-bench retail test tasks, never used for training, data selection or checkpoint choice
  (dev selection used 14 train-derived dev tasks; `tests/test_splits.py` checks the split is disjoint).
- **Trials:** 3 per task (seeds 300-302), greedy decoding, same user simulator and reward for every model.
- **Models:** B0 (base model, same prompt and format), the student (epoch3 adapter, frozen by sha256 before testing),
  the teacher (DeepSeek V4.1 Flash, `base` variant).
- **Statistics:** pass^k by the τ-bench estimator; 95% CIs by resampling tasks (2,000 draws); paired differences on
  the same tasks. Crashes and aborts score 0 (D10).
- **Pre-registration:** `results/threshold.md`, written before any student test episode.

## Results
See README "Result" and EXPERIMENT_LOG E5. Verdict: **bar met** (student − B0 pass^1 +37.5 pts, CI [+25.8, +48.3]).

## What the result does and does not show
- It shows that behaviour cloning on ~370 successful teacher trajectories turns a model that almost never completes a
  task (B0 2.5%) into one that completes 40% of held-out tasks on a single try, mainly by learning the tool-call
  protocol and the task flow (valid-call rate 7% → 89%).
- It does not show teacher-level reliability: pass^3 is 10% vs 72.5%. Ten of 40 tasks were never solved.
- It does not show a cost advantage under the serving setup used (see E5); that needs an efficient server.
- The student's records come from three machines with slightly different library versions (D11).

## Failure cases (student, test)
- `too_many_errors`: repeated invalid tool calls (e.g. searching for a product that does not exist, exchanging an
  order that is not delivered) until τ²'s 10-error limit.
- Episodes that end normally but fail the DB check (wrong or missing writes): the most common failure by count
  (108 of 120 student episodes ended normally, 48 succeeded); not yet broken down by cause.
- 3 crashes: GPU out-of-memory on very long conversations (T4, 15 GB).
