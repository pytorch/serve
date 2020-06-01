#!/usr/bin/env bash

set -eou pipefail

PYTHON_VERSIONS="3.6 3.7 3.8"
PKGS="torchserve torch-model-archiver"
TORCHSERVE_VERSION=$(cat ../../ts/version.txt)
TORCH_MODEL_ARCHIVER_VERSION=$(cat ../../model-archiver/model_archiver/version.txt)

for pkg in ${PKGS}; do
    PKG_VERSION=$(echo $pkg | tr 'a-z' 'A-Z' | tr '-' '_')_VERSION
    for python_version in ${PYTHON_VERSIONS}; do
        (
            set -x
            TORCHSERVE_VERSION=$TORCHSERVE_VERSION TORCH_MODEL_ARCHIVER_VERSION=$TORCH_MODEL_ARCHIVER_VERSION conda build --output-folder output/ --python="${python_version}" "${pkg}"
            set +x
        )
    done
    rm $pkg/meta.yaml
done
