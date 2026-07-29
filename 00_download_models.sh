#!/bin/bash
# Download Kimi-K3 (~1561 GB) + DSpark draft model (~4.5 GB) to local NVMe.
# Run on the HOST (uses /opt/pytorch venv), not inside the container.
#
#   bash 00_download_models.sh            # both models
#   bash 00_download_models.sh draft      # draft only (fast, for a quick smoke test)
#
# Both repos are public, so HF_TOKEN is optional. Set it if you hit rate limits.
set -euo pipefail

NVME="${NVME:-/opt/dlami/nvme}"
MODEL_DIR="${MODEL_DIR:-$NVME/models}"
HF_HOME_DIR="${HF_HOME_DIR:-$NVME/hf}"

MAIN_REPO="moonshotai/Kimi-K3"
DRAFT_REPO="RadixArk/Kimi-K3-DSpark"

source /opt/pytorch/bin/activate
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
# 96 shards @ ~17 GB; 16 workers saturates the instance's network without
# thrashing the NVMe.
export HF_HUB_DOWNLOAD_TIMEOUT=60

mkdir -p "$MODEL_DIR" "$HF_HOME_DIR"

dl() {
    local repo="$1" dest="$2" workers="$3"
    echo "==> $repo -> $dest"
    hf download "$repo" \
        --local-dir "$dest" \
        --max-workers "$workers"
}

case "${1:-all}" in
    draft)
        dl "$DRAFT_REPO" "$MODEL_DIR/Kimi-K3-DSpark" 8
        ;;
    main)
        dl "$MAIN_REPO" "$MODEL_DIR/Kimi-K3" 16
        ;;
    all)
        # Draft first: it is tiny and validates credentials/connectivity.
        dl "$DRAFT_REPO" "$MODEL_DIR/Kimi-K3-DSpark" 8
        dl "$MAIN_REPO" "$MODEL_DIR/Kimi-K3" 16
        ;;
    *)
        echo "usage: $0 [all|main|draft]" >&2; exit 1
        ;;
esac

echo
echo "==> done"
du -sh "$MODEL_DIR"/* 2>/dev/null || true
