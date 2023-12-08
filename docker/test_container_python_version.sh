#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

IMAGE_TAG=$1
EXPECTED_VERSION=$2

tmpfile=$(mktemp ./pyversionXXX.txt)
assert_py_version() {
    echo "Checking Python version..."
    docker run --rm -t "${IMAGE_TAG}" exec python --version > "${tmpfile}"
    if ! grep -q "Python ${EXPECTED_VERSION}" "${tmpfile}"
    then
        echo "Test failed: Wrong Python version. Expected ${EXPECTED_VERSION}, got $(cat "${tmpfile}")"
        exit 1
    else
        echo "Test succesful! Found version: $(cat "${tmpfile}")"
    fi
}

assert_py_version

rm -f "${tmpfile}"
