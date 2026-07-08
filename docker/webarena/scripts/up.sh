#!/usr/bin/env bash
# Bring the forum up and wait until it's serving. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || cp .env.example .env
docker compose up -d
./scripts/wait-ready.sh

echo
echo "Reddit is live. To point ModalityBench at it:"
echo "  source docker/webarena/scripts/print-env.sh   # exports REDDIT"
echo "  python docker/webarena/scripts/capture-auth.py # REQUIRED: all reddit tasks are login-gated"
echo "  python -m modalitybench.cli run configs/webarena-reddit.yaml"
