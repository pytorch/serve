#!/usr/bin/env bash

set -eou pipefail

cleanup()
{
  rm -rf dist
  rm -rf build
  rm -rf *.egg-info
}

create_torchserve_wheel()
{
  cd ../../

  python setup.py bdist_wheel --release --universal

  cp dist/*.whl binaries/pip/output/

  cleanup

  cd -
}

create_model_archiver_wheel()
{
  cd ../../model-archiver

  python setup.py bdist_wheel --release --universal

  cp dist/*.whl ../binaries/pip/output/

  cleanup

  cd -
}

rm -rf output

mkdir output

create_torchserve_wheel

create_model_archiver_wheel
