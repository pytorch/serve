#!/usr/bin/env bash

set -eou pipefail

PYTHON_VERSIONS="3.6 3.7 3.8"
PKGS="torchserve torch-model-archiver"

for pkg in ${PKGS}; do
    for python_version in ${PYTHON_VERSIONS}; do
        (
            set -x
            conda build --output-folder output/ --python="${python_version}" "${pkg}"
        )
    done
done
