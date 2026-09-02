#!/bin/bash
# Greedy probes for the v1-vs-v2 numerics comparison. $1 = port, $2 = tag
PORT=$1; TAG=$2
OUT=/opt/dlami/nvme/k3_probe_$TAG.txt
: > $OUT
i=0
while IFS= read -r q; do
  i=$((i+1))
  r=$(curl -s -m 180 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\":\"/models/Kimi-K3\",
    \"messages\":[{\"role\":\"user\",\"content\":$q}],
    \"temperature\":0,\"top_p\":1,\"max_tokens\":96,\"seed\":42}")
  echo "### Q$i $q" >> $OUT
  echo "$r" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    c=d[\"choices\"][0]
    print(\"FINISH:\",c[\"finish_reason\"],\"| tokens:\",d[\"usage\"][\"completion_tokens\"])
    print(c[\"message\"][\"content\"])
except Exception as e:
    print(\"PARSE-FAIL\",e); print(sys.stdin.read()[:400])
" >> $OUT
  echo >> $OUT
done <<QS
"What is the capital of France? Answer in one word."
"Compute 17 * 23. Show only the number."
"List the first 8 prime numbers, comma separated."
"Write one sentence explaining what a GPU is."
QS
cat $OUT
