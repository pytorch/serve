#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

IMAGE_TAG=$1
CONTAINER="test-container-py${IMAGE_TAG}"


healthcheck() {
    docker run -d --rm -it -p 8080:8080 --name="${CONTAINER}" "${IMAGE_TAG}"

    echo "Waiting 5s for container to come up..."
    sleep 5

    RESPONSE=$(curl localhost:8080/ping | jq .status)
    if [ "${RESPONSE}" == '"Healthy"' ]; then
        echo "Healthcheck succesful! Response from ${CONTAINER}: ${RESPONSE}"
    else
        echo "Healthcheck failed! Response from ${CONTAINER}: ${RESPONSE}"
        exit 1
    fi
}

healthcheck
docker stop "${CONTAINER}"
