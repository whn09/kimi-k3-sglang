#!/bin/bash
# First-token top-5 logprobs on a fixed prompt: the strongest cheap numerics check.
PORT=$1; TAG=$2
for p in "The capital of France is" "2 + 2 =" "def fibonacci(n):"; do
  curl -s -m 120 localhost:$PORT/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"/models/Kimi-K3\",\"prompt\":\"$p\",\"max_tokens\":1,\"temperature\":0,\"logprobs\":5}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
lp=d[\"choices\"][0][\"logprobs\"]
top=lp[\"top_logprobs\"][0]
print(\"PROMPT: $p\")
for k,v in sorted(top.items(), key=lambda kv:-kv[1]):
    print(\"   %-14r %+.6f\"%(k,v))
"
done | tee /opt/dlami/nvme/k3_lp_$TAG.txt
