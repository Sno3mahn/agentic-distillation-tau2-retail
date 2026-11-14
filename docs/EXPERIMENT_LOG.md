# Experiment Log

_Chronological. Numbers here come from saved records under `data/` (gitignored); commands are given so they can be regenerated._

## E1 — Teacher pilot, three variants (2026-11-14, slice S3)
**Question:** how often does the DeepSeek teacher succeed on retail train tasks, what does each variant cost, and which variant should generate the training data?

**Setup:** 10 train tasks (seeded sample, seed 20261114: 4 11 19 30 63 66 67 73 78 93) × 2 trials × 3 variants = 60 episodes. `deepseek-flash` agent and user simulator; user simulator temperature 0; trial seeds 300/301 shared across variants (paired). Reward = DB × substring COMMUNICATE (D4). Run off-peak (Saturday). Regenerate: `python -m agentic_distill.metrics base=data/raw/s3_pilot10_base rationale=data/raw/s3_pilot10_rationale thinking=data/raw/s3_pilot10_thinking`.
Variants: **base** non-thinking; **rationale** non-thinking + one-sentence rationale before each tool call; **thinking** DeepSeek thinking mode (reasoning not shown to the student).

| metric | base | rationale | thinking |
|---|---|---|---|
| episodes passed | 17/20 | 19/20 | 20/20 |
| pass^1 (95% CI over tasks) | 85% [70, 100] | 95% [85, 100] | 100% [100, 100] |
| pass^2 | 70% [40, 100] | 90% [70, 100] | 100% [100, 100] |
| valid-call rate | 96% | 96% | 97% |
| steps per episode | 12.7 | 12.9 | 12.4 |
| loop/abort rate | 0% | 0% | 0% |
| cost per episode / per success | $0.0020 / $0.0023 | $0.0020 / $0.0021 | $0.0031 / $0.0031 |
| failure causes | 2 COMMUNICATE, 1 DB | 1 COMMUNICATE | none |

Paired differences vs base (tasks in common, 95% CI): rationale pass^1 +10 pts [0, 25]; thinking pass^1 +15 pts [0, 30].
Rendered token lengths (student format): median ~9.0–9.2K, p90 ≤ 10.7K, max 10,851; none near the 32K limit. Combined with the earlier 2-task retries (base 5/8 including the first smoke, rationale 5/6, thinking 7/7 + 1/1) the direction is the same.

**Findings**
- Teacher success is high on these tasks (85–100%), so data yield is not a bottleneck. Every sampled task had at least one success in every variant.
- Differences between variants are not statistically distinguishable at this size: the paired CIs touch 0.
- Most failures were COMMUNICATE (the agent did not state a required figure), concentrated in tasks 4 and 19, not DB errors.
- Thinking costs ~1.55× per episode; 20/20 thinking episodes recorded reasoning tokens and 0/40 of the others did, so the mode switch works.
- Spend this experiment: ~$0.14. Slice `s3_pilot` total $0.1945 of $0.40; project total ≈ $0.20 of $5.

**Limitations:** 10 tasks, 2 trials (pass^3 not computable); single seed pair; substring scoring can give false passes (never false failures); user-simulator noise; a seeded sample of train tasks, not a ranking over the full set.

**Not decided here:** K, k, the variant for S4 data. See DECISIONS.md once chosen.

## E0 — B0 (base student) smoke runs (2026-11-14)
See DESIGN.md "S2b results" and "B0 under D7". Qualitative only (3 dev episodes, all reward 0).
