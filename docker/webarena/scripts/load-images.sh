#!/usr/bin/env bash
# Load the downloaded forum tar into Docker as the image the compose file references.
set -euo pipefail
cd "$(dirname "$0")/.."

TAR="downloads/postmill-populated-exposed-withimg.tar"
[ -f "$TAR" ] || { echo "Missing $TAR — run scripts/download-images.sh first" >&2; exit 1; }
echo "docker load < $TAR (this unpacks several GB) ..."
docker load --input "$TAR"
echo "Loaded. Image: postmill-populated-exposed-withimg"
