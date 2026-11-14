#!/bin/sh
# S5: push code bundle + SFT data as PRIVATE Kaggle datasets and the training notebook as a PRIVATE kernel.
# Usage: scripts/kaggle_push_s5.sh [--no-run]   (needs .venv/bin/kaggle and ~/.kaggle/access_token)
# The Kaggle username is read from the CLI config at run time, never stored in the repo.
set -e
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
U=$($K config view | awk '/username/{print $NF}')
[ -n "$U" ] || { echo "no kaggle username"; exit 1; }
push_ds() {  # name dir
  printf '{"title":"%s","id":"%s/%s","licenses":[{"name":"other"}]}\n' "$1" "$U" "$1" > "$2/dataset-metadata.json"
  if $K datasets status "$U/$1" >/dev/null 2>&1; then $K datasets version -p "$2" -m "$(git rev-parse --short HEAD)"
  else $K datasets create -p "$2"; fi        # private by default
}
scripts/make_kaggle_bundle.sh
DS=/tmp/kaggle_ds; SD=/tmp/kaggle_sft; KN=/tmp/kaggle_kernel_s5
rm -rf $DS $SD $KN; mkdir -p $DS $SD $KN
cp /tmp/agentic_distill_src.zip $DS/ && push_ds agentic-distill-src $DS
cp data/sft/train_v2.jsonl $SD/ && push_ds agentic-distill-sft $SD
cp notebooks/s5_train_kaggle.ipynb $KN/
printf '{"id":"%s/agentic-distill-s5-train","title":"agentic-distill-s5-train","code_file":"s5_train_kaggle.ipynb","language":"python","kernel_type":"notebook","is_private":true,"enable_gpu":true,"enable_internet":true,"dataset_sources":["%s/agentic-distill-src","%s/agentic-distill-sft"]}\n' "$U" "$U" "$U" > $KN/kernel-metadata.json
$K kernels push -p $KN --accelerator NvidiaTeslaT4 "$@"
