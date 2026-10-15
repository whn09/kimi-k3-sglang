#!/bin/bash
# HOST-side: build the image on THIS host and push it to ECR. Run on B300-1.
#
#   bash 01_build_push.sh                       # build + push $IMAGE
#   PUSH=0 bash 01_build_push.sh                # build only
#   DEEPEP_MAX_JOBS=16 bash 01_build_push.sh    # gentle on a host running someone's job
#
# Then on the other hosts: aws ecr get-login-password ... | docker login; docker pull $IMAGE
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

log="/tmp/k3s-build-${IMAGE_TAG}.log"
docker build --progress=plain \
    ${DEEPEP_MAX_JOBS:+--build-arg DEEPEP_MAX_JOBS=$DEEPEP_MAX_JOBS} \
    -t "$IMAGE" -f Dockerfile . 2>&1 | tee "$log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "FATAL: build failed, see $log"; exit 1; }

[[ "${PUSH:-1}" == 1 ]] || exit 0
aws ecr describe-repositories --region "$K3_REGION" --repository-names "$ECR_REPO" >/dev/null 2>&1 \
    || aws ecr create-repository --region "$K3_REGION" --repository-name "$ECR_REPO" >/dev/null
aws ecr get-login-password --region "$K3_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
docker push "$IMAGE"
echo "pushed $IMAGE  config=$(docker image inspect -f '{{.Id}}' "$IMAGE")"
