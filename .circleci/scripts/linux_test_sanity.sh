#!/bin/bash

source scripts/install_utils

MODELS=("fastrcnn" "fcn_resnet_101" "my_text_classifier" "resnet-18")
MODEL_INPUTS=("examples/object_detector/persons.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg")
HANDLERS=("object_detector" "image_segmenter" "text_classification" "image_classifier")

mkdir model_store

start_torchserve

for i in ${!MODELS[@]};
do
  model=${MODELS[$i]}
  input=${MODEL_INPUTS[$i]}
  handler=${HANDLERS[$i]}
  register_model "$model"
  run_inference "$model" "$input"
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