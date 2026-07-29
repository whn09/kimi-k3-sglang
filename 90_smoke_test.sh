#!/bin/bash
# Smoke test against a running endpoint.
#
#   bash 90_smoke_test.sh                 # standalone on localhost:30000
#   ENDPOINT=localhost:8080 bash 90_smoke_test.sh   # via the PD router
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ENDPOINT="${ENDPOINT:-localhost:$PORT}"

echo "=== /health ==="
curl -sf "http://${ENDPOINT}/health" && echo " OK" || echo " FAIL"

echo "=== /get_model_info ==="
curl -s "http://${ENDPOINT}/get_model_info" | python3 -m json.tool 2>/dev/null || echo "(no model info)"

echo
echo "=== chat completion ==="
curl -s "http://${ENDPOINT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is 127 * 34? Answer with just the number.\"}],
      \"max_tokens\": 256,
      \"temperature\": 0
    }" | python3 -m json.tool

echo
echo "=== streaming (10 chunks) ==="
curl -sN "http://${ENDPOINT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 20.\"}],
      \"max_tokens\": 128,
      \"stream\": true
    }" | head -10
