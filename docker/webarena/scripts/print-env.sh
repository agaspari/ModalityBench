# Emit the site URL ModalityBench's WebArenaSource reads. SOURCE this (don't execute) so REDDIT
# is exported into your shell:  source docker/webarena/scripts/print-env.sh
export REDDIT="${REDDIT:-http://localhost:${REDDIT_PORT:-9999}}"
echo "REDDIT=$REDDIT"
