#!/bin/sh
# Bundle the committed repo + the split file for upload as a PRIVATE Kaggle dataset.
# Usage: scripts/make_kaggle_bundle.sh   -> /tmp/agentic_distill_src.zip
set -e
cd "$(dirname "$0")/.."
git diff --quiet && git diff --cached --quiet || { echo "commit your changes first"; exit 1; }
rm -f /tmp/agentic_distill_src.zip
git archive --format=zip HEAD -o /tmp/agentic_distill_src.zip
zip -q /tmp/agentic_distill_src.zip data/splits/retail_split.json
echo "wrote /tmp/agentic_distill_src.zip ($(git rev-parse --short HEAD))"
