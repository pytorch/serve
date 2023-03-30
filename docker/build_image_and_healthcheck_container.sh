#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

VERSION=$1  # 3.8, 3.9 or 3.10
START_WAITING_TIME=2

IMAGE="test-image-${VERSION}"
CONTAINER="test-container-py${VERSION}"

echo "Building ${IMAGE}"
./build_image.sh -py "${VERSION}" -t "${IMAGE}"


healthcheck() {
    echo "Testing ${CONTAINER}..."
    docker run -d --rm -it -p 8080:8080 --name=${CONTAINER} $IMAGE > /dev/null 2>&1

    trap "echo stoping ${CONTAINER}; docker stop ${CONTAINER}" EXIT QUIT TERM

    echo "Waiting ${START_WAITING_TIME}s for container to come up..."
    sleep ${START_WAITING_TIME}

    RESPONSE=$(curl localhost:8080/ping | jq .status)
    if [ "${RESPONSE}" == '"Healthy"' ]; then
        echo "Healthcheck succesful! Response from ${CONTAINER}: ${RESPONSE}"
    else
        echo "Healthcheck failed! Response from ${CONTAINER}: ${RESPONSE}"
        exit 1
    fi
}

assert_py_version() {
    echo "Checking Python version..."
    tmpfile=$(mktemp /tmp/pyversion.XXXXXX.txt)
    tmp_container="test_container"
    docker run --rm -it $IMAGE exec python --name=${tmp_container} --version > $tmpfile
    docker stop ${tmp_container} 
    if ! grep -q "Python ${VERSION}" $tmpfile 
    then
        echo "Test failed: Wrong Python version. Expected ${VERSION}, got $(cat $tmpfile)"
        exit 1
    else
        echo "Test succesful! Found version: $(cat $tmpfile)"
    fi
}

healthcheck
assert_py_version

