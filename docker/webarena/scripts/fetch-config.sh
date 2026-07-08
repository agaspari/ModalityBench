#!/usr/bin/env bash
# Download WebArena's raw task config (all 812 tasks with __REDDIT__ etc. placeholders still in
# place). WebArenaSource substitutes the placeholders from env and filters to sites: [reddit],
# so there's no need to run WebArena's own generate_test_data.py.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p config_files
URL="https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
curl -L -o config_files/test.raw.json "$URL"
echo "saved -> config_files/test.raw.json"
