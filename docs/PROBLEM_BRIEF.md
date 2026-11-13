# Problem Brief — Agentic Distillation

_Stage 1 (problem exploration). Last updated 2026-11-11._

## Problem statement (organizer-stated)

> Agentic distillation: distill a stronger teacher's agent trajectories into a small student model that is cheap to run or runs locally.

This is Track 3 ("Agentic Fine-tuning"), problem statement 3, of the Lexsi AI Safety and Alignment Hackathon.

**In plain language:** have a strong model do multi-step tasks using tools, and record every step it takes. Then train a much smaller model on those records so it can do similar tasks nearly as well, at far lower cost.

## Key terms

- **Agent:** a language model working in a loop. It reads the situation, picks an action (often a tool call), sees the result, and repeats until it is done.
- **Trajectory:** the full record of one episode: the task, the reasoning steps, each tool call and its result, the final answer, and possibly whether it succeeded.
- **Teacher / student:** the stronger model being copied, and the smaller model being trained.
- **Distillation:** moving ability from a large model into a small one. The statement does not say how this should be done.

## Organizer-stated facts

Source: hackathon.lexsi.ai home page, read on 2026-11-11. The `/tracks` URL returned a 404 error, and the page was read through a summarizing tool, so the exact wording still needs checking on the site or Discord.

- **Suggested repos:** AgentTune and AuditKIT. The usual rule of using at least two track repos together does **not** apply to Track 3.
- **Student model:** must be Tiny Aya (Base, Global, Fire, Water or Earth) or Aya Expanse 8B/32B. Other models may only act as teacher, validator or judge.
- **Hosting:** public Hugging Face model and dataset collections, plus a public repo or demo.
- **Submission package:**
  - abstract under 500 words
  - write-up
  - GitHub repo with reproducibility notes
  - list of the repos, models, tracks and problem statements used
- **Required results:** before-and-after results using more than one metric, with reasons for the metric, dataset and model choices, and an explanation of why the new model is better to use.
- **Optional but preferred:** a cost (ROI) comparison between paying a closed model's API and self-hosting the small model.
- **Timeline:**
  - submissions due Nov 15, 23:59 AoE
  - finalists named Nov 16
  - finalist video or slides due Nov 22
- **Compute:** $20 of Hugging Face credits, plus free Colab and Kaggle quotas.
- **Judging:** detailed criteria and weightings are not published.

## User decisions

| Date | Decision | Status |
|---|---|---|
| 2026-11-11 | This is a real hackathon submission. The rules and the Nov 15 deadline are hard constraints. | Decided |
| 2026-11-11 | "Cheap to run" is the main story. "Runs locally" is a secondary goal, pursued only if feasible. | Decided |

## Open questions

- **What "distill trajectories" covers.** The narrow reading is that the student imitates saved teacher trajectories. The broad reading allows any teacher-to-student transfer built mainly on trajectories. The user is not sure yet; we will revisit this in domain learning (stage 3).

## Unknowns

- Task domain and the tools the agent can use
- Which teacher, how we access it, and the budget
- Student size: Tiny Aya or Aya Expanse 8B/32B
- Tiny Aya's parameter count (believed to be about 3B; not yet verified)
- Trajectory format
- Success metrics and baselines
- How much data
- Team, registration and compute status
- Whether AgentTune supports this workflow
