# DAgger round 2 (success-filtered labels): pre-registered evaluation plan

Written 2026-11-16 02:40 IST, before the round-2 training data exists in final form, before any round-2 adapter exists
and before any round-2 dev or test episode. **Post-hoc and exploratory**, like round 1 (`results/threshold_dagger.md`).
The main pre-registered result stays `results/threshold.md` (epoch3 vs B0); nothing here changes it.

## What changes versus round 1 (and nothing else)
Round 1 kept one teacher turn per student state without knowing whether it leads to success. Round 2 rolls the teacher
forward from each state to the end of the episode, K = 4 times (`src/agentic_distill/dagger_rollout.py`), scores the
full trajectory against the original task (DB x substring COMMUNICATE, from scratch), and keeps a label only if a
successful rollout starts with it. States with no successful rollout are dropped. Same student rollouts and states as
round 1 (task 59 excluded), same mix recipe (all labels + 25% replay of `train_v2.jsonl`), same hyperparameters
(2 epochs, lr 5e-5, seed 20261117, start from the frozen epoch3 adapter, not from round 1).

## Checkpoint
Fixed in advance: the final one (epoch 2 of this run). Its sha256 is appended below before any evaluation episode.

## Dev gate (decides whether to spend a test run)
Round 2, round 1 and epoch3 on all 14 dev tasks x 1 trial; round 2 and round 1 run in the same Modal setup as the
round-1 dev gate (epoch3 and round-1 dev records already exist from that setup and are reused). The test run for round 2
starts only if round-2 dev pass^1 >= round-1 dev pass^1 and >= epoch3 dev pass^1, and it can start before DeepSeek peak
hours (06:30 IST). With 14 episodes this only screens out a clear regression.

## Test comparison (if run)
40 test tasks x 3 trials, same settings as every other test run. Reported: round 2 vs epoch3 (primary), round 2 vs
round 1, round 2 vs B0, paired over tasks, 95% task-resampled bootstrap CIs. Round 2 is called better than epoch3 only
if the CI lower bound of (round 2 - epoch3) pass^1 is above 0. **Two DAgger variants are evaluated, so both are
reported, and the better-looking one is not selected after the fact:** the write-up's headline stays epoch3 vs B0.

## Known weaknesses
- The same ~50 states: this changes label quality, not the amount of data.
- Teacher rollouts that succeed can still start with a turn that is suboptimal for the student.
- The user simulator is stochastic, so a rollout's success has noise even for a good label (K = 4 reduces it).
- Environment confound with epoch3's pooled test records, as in round 1.

## Final checkpoint hash (appended after training, before any evaluation episode)
