#!/bin/bash
set -euxo pipefail

source scripts/install_utils

cleanup()
{
  stop_torchserve

  rm -rf model_store

  rm -rf logs

  # clean up residual from model-archiver IT suite.
  rm -rf model_archiver/model-archiver/htmlcov_ut model_archiver/model-archiver/htmlcov_it
}

install_pytest_suite_deps

install_bert_dependencies

run_backend_pytest

build_frontend

run_backend_python_linting

run_model_archiver_python_linting

run_model_archiver_UT_suite

./scripts/install_from_src

run_model_archiver_IT_suite

mkdir -p model_store

start_torchserve

models=("fastrcnn" "fcn_resnet_101" "my_text_classifier" "resnet-18" "my_text_classifier_scripted" "alexnet_scripted" "fcn_resnet_101_scripted"
          "roberta_qa_torchscript" "roberta_qa_no_torchscript" "bert_token_classification_torchscript" "bert_token_classification_no_torchscript"
          "bert_seqc_with_torchscript" "bert_seqc_without_torchscript")
model_inputs=("examples/object_detector/persons.jpg,docs/images/blank_image.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg"
 "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/Huggingface_Transformers/sample_text.txt" "examples/Huggingface_Transformers/sample_text.txt"
 "examples/Huggingface_Transformers/sample_text.txt" "examples/Huggingface_Transformers/sample_text.txt" "examples/Huggingface_Transformers/sample_text.txt" "examples/Huggingface_Transformers/sample_text.txt")
handlers=("object_detector" "image_segmenter" "text_classification" "image_classifier" "text_classification" "image_classifier" "image_segmenter" "custom" "custom" "custom" "custom" "custom" "custom")


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
