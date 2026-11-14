#!/bin/sh
# Push the committed repo as a PRIVATE Kaggle dataset and the notebook as a PRIVATE kernel.
# Usage: scripts/kaggle_push.sh [--no-run]     (needs .venv/bin/kaggle and ~/.kaggle/access_token)
# The Kaggle username is read from the CLI config at run time, never stored in the repo.
set -e
cd "$(dirname "$0")/.."
K=.venv/bin/kaggle
U=$($K config view | awk '/username/{print $NF}')
[ -n "$U" ] || { echo "no kaggle username"; exit 1; }
scripts/make_kaggle_bundle.sh
DS=/tmp/kaggle_ds; KN=/tmp/kaggle_kernel
rm -rf $DS $KN; mkdir -p $DS $KN
cp /tmp/agentic_distill_src.zip $DS/
printf '{"title":"agentic-distill-src","id":"%s/agentic-distill-src","licenses":[{"name":"other"}]}\n' "$U" > $DS/dataset-metadata.json
if $K datasets status "$U/agentic-distill-src" >/dev/null 2>&1; then
  $K datasets version -p $DS -m "bundle $(git rev-parse --short HEAD)"
else
  $K datasets create -p $DS            # private by default
fi
cp notebooks/s2b_b0_kaggle.ipynb $KN/
printf '{"id":"%s/agentic-distill-s2b-b0","title":"agentic-distill-s2b-b0","code_file":"s2b_b0_kaggle.ipynb","language":"python","kernel_type":"notebook","is_private":true,"enable_gpu":true,"enable_internet":true,"dataset_sources":["%s/agentic-distill-src"]}\n' "$U" "$U" > $KN/kernel-metadata.json
$K kernels push -p $KN --accelerator NvidiaTeslaT4 "$@"
