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

run_backend_pytest

run_backend_python_linting

run_model_archiver_UT_suite

./scripts/install_from_src_ubuntu

run_model_archiver_IT_suite

mkdir -p model_store

start_torchserve


models=("fastrcnn" "fcn_resnet_101" "my_text_classifier" "resnet-18")
model_inputs=("examples/object_detector/persons.jpg,docs/images/blank_image.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg")
handlers=("object_detector" "image_segmenter" "text_classification" "image_classifier")

for i in ${!models[@]};
do
  model=${models[$i]}
  inputs=$(echo ${model_inputs[$i]} | tr "," "\n")
  handler=${handlers[$i]}
  register_model "$model"
  for input in ${inputs[@]};
  do
    run_inference "$model" "$input"
  done
  #skip unregistering resnet-18 model to test snapshot feature with restart
  if [ "$model" != "resnet-18" ]
  then
    unregister_model "$model"
  fi
  echo "$handler default handler is stable."
done

stop_torchserve

# restarting torchserve
# this should restart with the generated snapshot and resnet-18 model should be automatically registered

start_torchserve

run_inference resnet-18 examples/image_classifier/kitten.jpg

stop_torchserve

cleanup

echo "CONGRATULATIONS!!! YOUR BRANCH IS IN STABLE STATE"
