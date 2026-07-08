#!/usr/bin/env bash
# Poll the forum until it answers, so callers don't race the container's startup.
set -euo pipefail
URL="${1:-http://localhost:${REDDIT_PORT:-9999}}"

echo "Waiting for forum at $URL ..."
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$URL" || true)
  case "$code" in
    200|301|302) echo "forum is up (HTTP $code)"; exit 0 ;;
  esac
  sleep 2
done
echo "forum did not become ready within ~2 min" >&2
exit 1
