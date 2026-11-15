#!/usr/bin/env bash
# DAgger round 2, end to end, applying the rules in results/threshold_dagger2.md without a human in the loop:
# wait for the relabel -> train (Modal L4) -> append adapter hash + commit -> dev gate (Modal T4) -> test if the gate passes.
set -euo pipefail
cd "$(dirname "$0")/.."
M=.venv/bin/modal
log() { echo "[$(date '+%H:%M:%S')] $*"; }

while pgrep -f agentic_distill.dagger_rollout >/dev/null; do sleep 30; done
[ -s data/sft/dagger_r2.jsonl ] || { log "FAILED: no data/sft/dagger_r2.jsonl"; exit 1; }
log "relabel done: $(wc -l < data/sft/dagger_r2.jsonl) examples"

$M run --detach scripts/modal_dagger_train.py --dagger data/sft/dagger_r2.jsonl --tag dagger_r2
A=data/modal_out_dagger_r2_train/epoch2
[ -s $A/adapter_model.safetensors ] || { log "FAILED: training produced no epoch2 adapter"; exit 1; }
H=$(shasum -a 256 $A/adapter_model.safetensors | cut -d' ' -f1)
printf -- '- `dagger_r2_s20261117/epoch2/adapter_model.safetensors` sha256 `%s`\n  (recorded %s, after training and before any round-2 dev or test episode).\n' "$H" "$(date '+%Y-%m-%d %H:%M %Z')" >> results/threshold_dagger2.md
git add results/threshold_dagger2.md data/sft/dagger_r2_mix.jsonl 2>/dev/null || git add results/threshold_dagger2.md
git commit -q -m "DAgger r2: record final adapter hash before any eval episode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main || log "WARN: commit/push failed"
log "trained; sha256 $H"

$M run --detach scripts/modal_student_eval.py --split dev --tasks all --trials 1 --per-container 1 --adapter $A \
    --adapter-name dagger_r2_epoch2 --label s3-dagger-r2 --slice eval_student --out data/modal_out_daggerdev_r2
read -r N W < <(python3 - <<'P'
import json, glob
recs = {}
for f in glob.glob("data/modal_out_daggerdev_r2/g*/data/raw/*/episodes.jsonl"):
    for l in open(f):
        r = json.loads(l); recs[str(r["task_id"])] = r["reward"] > 0.5
print(len(recs), sum(recs.values()))
P
)
log "dev: round2 $W/$N (round1 3/14, epoch3 1/14)"
[ "$N" -eq 14 ] || { log "STOP: dev incomplete ($N/14), gate not evaluated"; exit 1; }
NOW=$(date +%H%M)
if [ "$W" -ge 3 ] && [ "$NOW" -lt 0630 ]; then
  log "gate PASS; launching test"
  $M run --detach scripts/modal_student_eval.py --split test --tasks all --trials 3 --per-container 2 --adapter $A \
      --adapter-name dagger_r2_epoch2 --label s3-dagger-r2 --slice eval_student --out data/modal_out_daggertest_r2
  log "test done: $(cat data/modal_out_daggertest_r2/g*/data/raw/*/episodes.jsonl | wc -l) episodes"
else
  log "gate FAIL or past 06:30 (wins=$W, time=$NOW): no test run"
fi
