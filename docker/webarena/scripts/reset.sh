#!/usr/bin/env bash
# Restore pristine site state between eval runs by recreating the container from the image.
# The Postmill image ships pre-populated, so a fresh container is a fresh, seeded site — this
# is how WebArena resets state for tasks that mutate it (posting, voting, editing).
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose down
docker compose up -d --force-recreate
./scripts/wait-ready.sh
echo "forum reset to pristine state"
