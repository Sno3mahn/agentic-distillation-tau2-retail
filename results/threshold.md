# Success threshold (C9 / AC10) — written 2026-11-15 13:49 IST, BEFORE any test-split run

No test-split episode (student, B0 or teacher) has been run when this file is created. The test runners refuse
to start unless this file exists.

## Frozen before testing
- **Student:** Tiny Aya En-Thinker + LoRA adapter `epoch3` (r=16, alpha=32, 3 epochs on `data/sft/train_v2.jsonl`,
  370 examples), adapter sha256 `88544be6d8a37ece4504ed88057c5bac148847def82e2bfc7ea548bf8a9c85b1`.
- **Chosen on dev:** epoch3 5/12 vs epoch2 3/12 episodes (dev tasks 103 and 110 crashed with GPU out-of-memory in both
  runs and were not scored). The gap is 2 episodes, so this choice is weakly supported; it is also the checkpoint with
  the lower training loss.
- **Decoding / protocol:** greedy, max_new_tokens 1024, D3.6 retry-then-abort, D7 stop at `<tool_result`; the same
  settings for B0. DeepSeek V4.1-Flash user simulator, temperature 0. Reward = DB x substring COMMUNICATE (D4).
- **Test set:** the 40 test tasks (split sha256 prefix `e13c96f12613`), 3 trials per task (seeds 300-302).
- **Failures:** an abort or a crash (e.g. out-of-memory) scores 0 and stays in the denominator.
- **Baselines:** B0 = the same model without the adapter, same prompt format. Teacher = DeepSeek V4.1-Flash, `base`
  variant (non-thinking, no rationale prompt), same user simulator and reward; reported for context, not as a bar.

## The bar (user-delegated; proposed by Claude 2026-11-15)
**The distillation counts as successful on the primary metric if** the student's test pass^1 exceeds B0's, with the
lower end of the paired 95% bootstrap CI (resampling tasks) of the difference above 0.

This is a bar of *improvement over the untrained student*, not of parity with the teacher. Nothing here claims the
student matches the teacher; the teacher's number is reported so the gap is visible.

## Reported regardless of outcome
pass^1 and pass^3 (with CIs), valid-call rate, steps per episode, loop/abort rate, crash rate, cost per success,
every failure category, and any run that crashed. If the bar is missed, the write-up says so.

## Known weaknesses in the protocol
- 3 trials per task and one seed per trial: the user simulator is noisy (e.g. dev behaviour of task 59-like ambiguities).
- Substring COMMUNICATE can pass a wrong statement of the required figure; no semantic audit was done.
- Dev selection used 12 single-trial episodes per checkpoint.
