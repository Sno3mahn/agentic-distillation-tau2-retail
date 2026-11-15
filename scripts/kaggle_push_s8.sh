#!/bin/sh
# Option-1 experiment: push code bundle + SFT data (PRIVATE datasets) and the continued-training notebook (PRIVATE kernel).
# Usage: scripts/kaggle_push_s8.sh [--no-run]
set -e
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
U=$($K config view | awk '/username/{print $NF}')
[ -n "$U" ] || { echo "no kaggle username"; exit 1; }
scripts/make_kaggle_bundle.sh
DS=/tmp/kaggle_ds; KN=/tmp/kaggle_kernel_s8
rm -rf $DS $KN; mkdir -p $DS $KN
cp /tmp/agentic_distill_src.zip $DS/
printf '{"title":"agentic-distill-src","id":"%s/agentic-distill-src","licenses":[{"name":"other"}]}\n' "$U" > $DS/dataset-metadata.json
$K datasets version -p $DS -m "$(git rev-parse --short HEAD)"
cp notebooks/s8_train_more_kaggle.ipynb $KN/
printf '{"id":"%s/agentic-distill-s8-train-more","title":"agentic-distill-s8-train-more","code_file":"s8_train_more_kaggle.ipynb","language":"python","kernel_type":"notebook","is_private":true,"enable_gpu":true,"enable_internet":true,"dataset_sources":["%s/agentic-distill-src","%s/agentic-distill-sft"],"kernel_sources":["%s/agentic-distill-s5-train"]}\n' "$U" "$U" "$U" "$U" > $KN/kernel-metadata.json
$K kernels push -p $KN --accelerator NvidiaTeslaT4 "$@"
