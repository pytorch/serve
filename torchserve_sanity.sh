#!/bin/bash
set -euxo pipefail

source scripts/install_utils

cleanup()
{
  stop_torchserve

  rm -rf model_store

  rm -rf logs
}

install_pytest_suite_deps

run_backend_python_linting

run_model_archiver_UT_suite

./scripts/install_from_src_ubuntu

run_model_archiver_IT_suite

mkdir -p model_store

start_torchserve

register_model resnet-18 resnet-18.mar

stop_torchserve

# restarting torchserve
# this should restart with the generated snapshot and resnet-18 model should be automatically registered

start_torchserve

run_inference resnet-18 examples/image_classifier/kitten.jpg

cleanup

echo "CONGRATULATIONS!!! YOUR BRANCH IS IN STABLE STATE"
