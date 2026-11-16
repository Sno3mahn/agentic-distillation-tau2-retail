# Agentic distillation on τ²-bench retail: DeepSeek V4.1 Flash → Tiny Aya En-Thinker (3.35B, LoRA)

Lexsi AI Safety & Alignment Hackathon, Track 3, problem statement 3 (agentic distillation).

A small open model (Tiny Aya En-Thinker, 3.35B) is taught to act as a multi-turn tool-calling customer-service agent by
**behaviour cloning** successful trajectories of a strong API teacher (DeepSeek V4.1 Flash) in the τ²-bench retail
environment, then evaluated on held-out tasks against the same model without training (B0) and against the teacher.

## Result (test split: 40 held-out tasks × 3 trials, pre-registered)

| model | pass^1 | pass^3 | valid tool-call rate | loop/abort rate | crash rate |
|---|---|---|---|---|---|
| B0: Tiny Aya En-Thinker, prompted | 2.5% [0.0, 7.5] | 2.5% [0.0, 7.5] | 7.3% | 30.8% | 0% |
| **Student: + LoRA SFT on teacher trajectories** | **40.0% [30.0, 50.0]** | **10.0% [2.5, 20.0]** | **88.7%** | **10.0%** | 2.5% |
| Teacher: DeepSeek V4.1 Flash (reference) | 89.2% [82.5, 95.0] | 72.5% [57.5, 85.0] | 97.7% | 0% | 0% |

- **Student − B0, paired over tasks: pass^1 +37.5 points, 95% CI [+25.8, +48.3].** The bar written in
  [`results/threshold.md`](results/threshold.md) before any student test episode (CI lower bound > 0) is **met**.
  With crashed episodes excluded instead of scored 0: +38.7 [+27.1, +50.0].
- Reliability is the gap: pass^3 (all 3 trials succeed) is 10% for the student vs 72.5% for the teacher.
- 95% CIs resample tasks, not episodes. Reproduce the table: `PYTHONPATH=src python scripts/pool_test.py`.
- **Cost (honest negative):** with plain HF `generate` on a T4 (~$0.59/h), the student costs **~$0.12 per successful
  task** vs **$0.0023** for the teacher API. Each turn re-processes the whole conversation (median 132K input tokens per
  episode, ~5 min/episode); serving with prefix caching and batching (e.g. vLLM) is the obvious next step and was
  not done here. The student's licence (CC-BY-NC-4.0) also rules out commercial use without a Cohere licence.

**Exploratory follow-ups (plans fixed in `results/threshold_dagger.md`, `results/threshold_dagger2.md`; the headline above stays epoch3 vs B0):**
two more epochs of SFT (cumulative 5) did not improve dev pass rate (3/8 vs 4/8 for epoch 3). One round of DAgger
(teacher labels at states the student itself visited) was tried in two variants, same 40 test tasks × 3 trials:

| model | pass^1 | pass^3 |
|---|---|---|
| epoch3 (headline student) | 40.0% [30.0, 50.0] | 10.0% [2.5, 20.0] |
| DAgger r1: one teacher turn per state, not success-filtered | 36.7% [25.8, 48.3] | 17.5% [5.0, 30.0] |
| DAgger r2: labels kept only if a teacher rollout from that state succeeds | 46.7% [33.3, 60.0] | 30.0% [17.5, 45.0] |

Paired vs epoch3: r1 −3.3 pts pass^1 [−13.3, +7.5]; r2 +6.7 pts pass^1 [−5.0, +18.3] and +20.0 pts pass^3 [+5.0, +37.5].
**Neither variant met the pre-registered rule (pass^1 CI lower bound > 0).** r2's pass^3 gain excludes 0, but it is a
secondary metric picked after seeing results, among several comparisons, so treat it as suggestive. See E7/E8 in
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md). Reproduce: `PYTHONPATH=src python scripts/pool_dagger.py test2`.

## Published artifacts

- Model (LoRA adapter, tag `v1-epoch3`): https://huggingface.co/ritabrata/tiny-aya-en-thinker-tau2-retail-lora
- Dataset (split, teacher traces, SFT set, every evaluation record and trace, tag `v1`): https://huggingface.co/datasets/ritabrata/agentic-distillation-tau2-retail-data
- Collections: [models](https://huggingface.co/collections/ritabrata/agentic-distillation-retail-models-6afa0df9dbeb2613fc80c373), [datasets](https://huggingface.co/collections/ritabrata/agentic-distillation-retail-datasets-6afa0dfb6a5b1af4d9af1ef8)
- Abstract and submission sheet: [`docs/SUBMISSION.md`](docs/SUBMISSION.md)

## Setup

| | |
|---|---|
| Environment | τ²-bench retail, commit `c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4` (MIT) |
| Split | official 74 train / 40 test; 14 dev tasks held out of train by a seeded shuffle → 60 / 14 / 40 ([`src/agentic_distill/splits.py`](src/agentic_distill/splits.py), [`tests/test_splits.py`](tests/test_splits.py)) |
| Reward | τ² DB-state check × substring COMMUNICATE check (no LLM judge) |
| User simulator | DeepSeek V4.1 Flash, temperature 0, for every model |
| Teacher data | 60 train tasks × 6 episodes (non-thinking + one-sentence rationale), plus a thinking-mode rerun on 17 hard tasks; success-filtered (reward = 1), ≤ 8 per task → **370 trajectories, 59 tasks** |
| Student | `CohereLabs/tiny-aya-en-thinker`, LoRA r=16, α=32 on all attention + MLP projections, 3 epochs, lr 1e-4, fp16, loss only on assistant turns |
| Format | tool calls as JSON in `<tool_call>` tags, tool results in `<tool_result>` blocks ([`formatter.py`](src/agentic_distill/formatter.py), shared by training and inference) |
| Decoding | greedy, max 1,024 new tokens per turn; an unparseable turn gets one format-error message, a second ends the episode (abort = failure) |

Design decisions and evidence: [`docs/DECISIONS.md`](docs/DECISIONS.md), experiments: [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md),
design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Reproduce

```bash
# CPU side (teacher runs, data building, metrics); Python 3.12-3.13
uv venv && source .venv/bin/activate
git clone https://github.com/sierra-research/tau2-bench third_party/tau2-bench
git -C third_party/tau2-bench checkout c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4
uv pip install -e third_party/tau2-bench -e ".[dev,student]"
export API_DEEPSEEK=...  HF_TOKEN=...          # HF token needs access to the gated base model
pytest -q                                      # formatter, loss masks, loss equivalence, metrics, splits

python -m agentic_distill.splits                                         # data/splits/retail_split.json
python -m agentic_distill.teacher_run --tasks all --trials 6 --slice s4_teacher_data --variant rationale
python -m agentic_distill.build_sft "data/raw/s4_rationale_p*" "data/raw/s4_thinkrat_h*" --cap-per-task 8 --out data/sft/train_v2.jsonl

# GPU side (one T4 is enough); pins in requirements-gpu.txt
python -m agentic_distill.train_lora --data data/sft/train_v2.jsonl --out out/lora --epochs 3
python -m agentic_distill.student_run --split test --tasks all --trials 3 --slice eval_student --label s2-epoch3 --adapter out/lora/epoch3
python -m agentic_distill.student_run --split test --tasks all --trials 3 --slice eval_student --label b0
python -m agentic_distill.teacher_run --split test --tasks all --trials 3 --slice eval_teacher --variant base
```

The test split is locked in code until `results/threshold.md` exists. The runners refuse to start in DeepSeek peak
hours (`--allow-peak` overrides) and stop at the per-slice budget in `configs/run.yaml`.

**From the published records instead:** download the dataset repo (below) into `data/` (it mirrors the paths used here)
and run `PYTHONPATH=src python scripts/pool_test.py`.

## Where each result came from

| result | script / notebook | hardware | libraries |
|---|---|---|---|
| teacher data, teacher test baseline | `teacher_run.py` | laptop (API only) | — |
| LoRA training (epochs 1-3) | `notebooks/s5_train_kaggle.ipynb` | Kaggle T4 | torch 2.11.0, transformers 5.23.0, peft 0.21.3 |
| dev evaluation, checkpoint choice | `notebooks/s6_dev_eval_kaggle.ipynb` | Kaggle T4 ×2 | Kaggle image, same day |
| student test, odd 20 tasks | `notebooks/s7_test_eval_kaggle.ipynb` | Kaggle T4 ×2 | Kaggle image, same day |
| student test, even tasks (chunks 0-2) | `notebooks/s7b_test_eval_colab.ipynb` | Colab T4 | torch 2.14.1, transformers 5.23.0, peft 0.21.3 |
| student test, even tasks (chunks 3-4) | `scripts/modal_s7b_missing.py` | Modal T4 | torch 2.15.0, transformers 5.23.0, peft 0.21.3 |
| B0 test | `notebooks/s7_b0_test_eval_kaggle.ipynb` | Kaggle T4 ×2 | transformers 5.23.0 |
| continued training, its dev eval | `notebooks/s8_train_more_kaggle.ipynb`, `scripts/modal_s8_dev.py` | Kaggle T4, Modal T4 | as above |
| DAgger rounds 1 and 2 | `dagger_relabel.py` (r1), `dagger_rollout.py` (r2), `scripts/modal_student_eval.py`, `scripts/modal_dagger_train.py`, `scripts/run_dagger_r2_chain.sh` | Modal T4 / L4 | `requirements-gpu.txt` |

`scripts/modal_student_eval.py` is the general runner; `modal_s7b_missing.py` and `modal_s8_dev.py` are kept as the
exact code that produced those records. The test set was split across machines because a Kaggle bug (fixed) lost
half of the first run and Colab's quota ran out (D11); greedy outputs can differ slightly across library versions.

## Limitations

- 40 test tasks × 3 trials: CIs are wide (±10 points on pass^1).
- Substring COMMUNICATE scoring can pass a reply that states a required figure wrongly; no semantic audit was done.
- The user simulator is an LLM: some tasks are ambiguous (task 59 in train: the simulator and the gold answer disagree).
- The student's test records come from three machines with slightly different library versions (D11).
- 3 of 120 student test episodes crashed with GPU out-of-memory on very long contexts (scored 0; sensitivity above).
- One student, one environment, one teacher; English only.

## Licences

- Code in this repository: MIT.
- τ²-bench: MIT. DeepSeek API outputs: the provider's terms assign output rights to the user and permit distillation.
- The adapter inherits the base model's **CC-BY-NC-4.0**; the base model (`CohereLabs/tiny-aya-en-thinker`) is gated on Hugging Face.
