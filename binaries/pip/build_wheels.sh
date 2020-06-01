#!/usr/bin/env bash

set -eou pipefail

source ../../scripts/install_utils

cleanup()
{
  echo "cleaning up residuals"
  rm -rf dist
  rm -rf build
  rm -rf *.egg-info
}

create_torchserve_wheel()
{

  cd ../../

  trap 'cleanup;exit 1' SIGINT SIGTERM EXIT

  python setup.py bdist_wheel --release --universal

  cp dist/*.whl binaries/pip/output/

  cleanup

  cd -

  trap - SIGINT SIGTERM EXIT
}

create_model_archiver_wheel()
{
  cd ../../model-archiver

  trap 'cleanup;exit 1' SIGINT SIGTERM EXIT

  python setup.py bdist_wheel --release --universal

  cp dist/*.whl ../binaries/pip/output/

  echo 'sleeping for test'

  sleep 60

  cleanup

  cd -

  trap - SIGINT SIGTERM EXIT
}

rm -rf output

mkdir output

install_java_deps

install_torch_deps

create_torchserve_wheel

create_model_archiver_wheel
