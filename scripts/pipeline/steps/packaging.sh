#!/usr/bin/env bash

set -e # exit immediately after any error

echo "
#############################################################################################################
### STEP : Packaging
###
### - Build and Push the Docker image
### - Export the IMAGE_DIGEST to be injected in k8s manifests
#############################################################################################################
"

#
# 1. Prepare Docker --build-arg arguments
#
DOCKER_BUILD_ARGS="--build-arg BUILD_VERSION=${GIT_SHORT_COMMIT}"

#
# 2. Build + Push docker image, then export IMAGE_DIGEST
#
. "${WORKDIR}/secure-pipeline-scripts/packaging/docker.sh" "${IMAGE_TAG}" "deploy/docker/hubserving/pretrain-hub/Dockerfile" "./" "${DOCKER_BUILD_ARGS}"
export IMAGE_DIGEST=${IMAGE_DIGEST}

