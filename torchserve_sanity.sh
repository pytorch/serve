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

# run object detection example

register_model "fastrcnn"

run_inference "fastrcnn" "examples/object_detector/persons.jpg"

unregister_model "fastrcnn"

echo "object_detector default handler is stable."

# run image segmentation example

register_model "fcn_resnet_101"

run_inference "fcn_resnet_101" "examples/image_segmenter/fcn/persons.jpg"

unregister_model "fcn_resnet_101"

echo "image_segmenter default handler is stable."

# run text classification example -

register_model "my_text_classifier"

run_inference "my_text_classifier" "examples/text_classification/sample_text.txt"

unregister_model "my_text_classifier"

echo "text_classifier default handler is stable."

# run image classification example

register_model "resnet-18"

run_inference "resnet-18" "examples/image_classifier/kitten.jpg"

echo "image_classifier default handler is stable."

stop_torchserve

# restarting torchserve
# this should restart with the generated snapshot and resnet-18 model should be automatically registered

start_torchserve

run_inference "resnet-18" "examples/image_classifier/kitten.jpg"

stop_torchserve

cleanup

echo "CONGRATULATIONS!!! YOUR BRANCH IS IN STABLE STATE"
