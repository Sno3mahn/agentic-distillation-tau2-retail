# DAgger round 1: pre-registered evaluation plan

Written 2026-11-15 23:16 IST, before any DAgger adapter exists and before any DAgger dev or test episode.
This is a **post-hoc, exploratory** experiment. The main, pre-registered result stays `results/threshold.md`
(epoch3 vs B0). Nothing here changes that file or its verdict.

## Model
- DAgger round 1 = the frozen epoch3 adapter (sha256 `88544be6…c85b1`) trained further on teacher labels at
  states the epoch3 student visited on the 60 **train** tasks (`src/agentic_distill/dagger_relabel.py`), plus a
  25% replay sample of the original SFT set. 2 epochs, lr 5e-5, seed 20261117 (`scripts/modal_dagger_train.py`).
- **The checkpoint is fixed in advance: the final one (epoch 2 of this run).** Its sha256 is appended below before
  any test episode. No choice between checkpoints is made on test.

## Dev gate (decides whether to spend a test run)
- DAgger and epoch3 both run on all 14 dev tasks x 1 trial, **in the same Modal environment**, same settings
  (greedy, max_new_tokens 1024, user simulator DeepSeek Flash at temperature 0).
- Test runs only if DAgger dev pass^1 >= epoch3 dev pass^1 and the run can start before DeepSeek peak hours.
- With 14 episodes this gate only screens out a clear regression; it is not evidence of improvement.

## Test comparison (if run)
- 40 test tasks x 3 trials, same settings as the epoch3 test run; crashes and aborts score 0.
- Reported: DAgger vs epoch3 and DAgger vs B0, paired over tasks, 95% task-resampled bootstrap CIs.
- DAgger is called better than epoch3 only if the paired CI lower bound of (DAgger - epoch3) pass^1 is above 0.
  Otherwise the write-up says the round did not show an improvement. Both rows are reported either way.

## Known weaknesses
- Environment confound: the epoch3 test records come from Kaggle, Colab and Modal; DAgger's would all come from Modal.
- Teacher labels for single turns are not success-filtered.
- One DAgger round, about 100-200 new states.

## Final checkpoint hash (appended after training, before any test episode)
- `dagger_r1_s20261117/epoch2/adapter_model.safetensors` sha256 `66eef03259d175dc13d220d0586f82356d166a0b627c3f697361d4b67f207dc1`
  (recorded 2026-11-16 02:01 IST, after training and before any DAgger dev or test episode).
