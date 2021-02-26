#!/usr/bin/env bash

set -eou pipefail

TOP_DIR=$(git rev-parse --show-toplevel)

TORCHSERVE_VERSION=${TORCHSERVE_VERSION:-$(cat ${TOP_DIR}/ts/version.txt)}
STAGING_ORG=${STAGING_ORG:-geeta}
PROMOTE_ORG=${PROMOTE_ORG:-pytorch}

DRY_RUN=${DRY_RUN:-enabled}
DOCKER="echo docker"
if [[ ${DRY_RUN} = "disabled" ]]; then
    DOCKER="docker"
    set -x
else
    echo "WARNING: DRY_RUN enabled, not doing any work"
fi


(
    ${DOCKER} pull "${STAGING_ORG}/torchserve:${TORCHSERVE_VERSION}-gpu"
    ${DOCKER} pull "${STAGING_ORG}/torchserve:${TORCHSERVE_VERSION}-cpu"
    ${DOCKER} pull "${STAGING_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}-gpu"
    ${DOCKER} pull "${STAGING_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}-cpu"

    ${DOCKER} tag "${STAGING_ORG}/torchserve:${TORCHSERVE_VERSION}-gpu"     "${PROMOTE_ORG}/torchserve:${TORCHSERVE_VERSION}-gpu"
    ${DOCKER} tag "${STAGING_ORG}/torchserve:${TORCHSERVE_VERSION}-gpu"     "${PROMOTE_ORG}/torchserve:latest-gpu"
    ${DOCKER} tag "${STAGING_ORG}/torchserve:${TORCHSERVE_VERSION}-cpu"     "${PROMOTE_ORG}/torchserve:${TORCHSERVE_VERSION}-cpu"
    ${DOCKER} tag "${STAGING_ORG}/torchserve:${TORCHSERVE_VERSION}-cpu"     "${PROMOTE_ORG}/torchserve:latest"
    ${DOCKER} tag "${STAGING_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}-gpu" "${PROMOTE_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}-gpu"
    ${DOCKER} tag "${STAGING_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}-cpu" "${PROMOTE_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}"

    ${DOCKER} push "${PROMOTE_ORG}/torchserve:${TORCHSERVE_VERSION}-gpu"
    ${DOCKER} push "${PROMOTE_ORG}/torchserve:latest-gpu"
    ${DOCKER} push "${PROMOTE_ORG}/torchserve:${TORCHSERVE_VERSION}-cpu"
    ${DOCKER} push "${PROMOTE_ORG}/torchserve:latest"
    ${DOCKER} push "${PROMOTE_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}-gpu"
    ${DOCKER} push "${PROMOTE_ORG}/torchserve-kfs:${TORCHSERVE_VERSION}"
)
