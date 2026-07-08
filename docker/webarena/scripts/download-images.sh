#!/usr/bin/env bash
# Download the Reddit (Postmill) image tar from WebArena's CMU mirror (no Google Drive gate).
# Resumable (-C -): re-run if the download is interrupted. A few GB — grab a coffee.
set -euo pipefail
cd "$(dirname "$0")/.."

FORUM_TAR="postmill-populated-exposed-withimg.tar"
URL="${FORUM_URL:-http://metis.lti.cs.cmu.edu/webarena-images/$FORUM_TAR}"

mkdir -p downloads
echo "Downloading $FORUM_TAR from $URL"
curl -L -C - -o "downloads/$FORUM_TAR" "$URL"
echo "Done -> downloads/$FORUM_TAR"
