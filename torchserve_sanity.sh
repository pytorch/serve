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

set +u
install_torch_deps $1
set -u

install_pytest_suite_deps

install_bert_dependencies

run_backend_pytest

build_frontend

run_backend_python_linting

run_model_archiver_python_linting

run_model_archiver_UT_suite

set +u
./scripts/install_from_src $1
set -u

if is_gpu_instance;
then
    cuda_status=$(python -c "import torch; print(int(torch.cuda.is_available()))")
    if [ $cuda_status -eq 0 ] ;
    then
      echo Ohh Its NOT running on GPU!!
      exit 1
    fi
fi

run_model_archiver_IT_suite

mkdir -p model_store

start_torchserve

models=("fastrcnn" "fcn_resnet_101" "my_text_classifier_v2" "resnet-18" "my_text_classifier_scripted_v2" "alexnet_scripted" "fcn_resnet_101_scripted"
           "roberta_qa_no_torchscript" "bert_token_classification_no_torchscript" "bert_seqc_without_torchscript")

model_inputs=("examples/object_detector/persons.jpg,docs/images/blank_image.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg"
 "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/Huggingface_Transformers/QA_artifacts/sample_text.txt"
 "examples/Huggingface_Transformers/Token_classification_artifacts/sample_text.txt" "examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt")
handlers=("object_detector" "image_segmenter" "text_classification" "image_classifier" "text_classification" "image_classifier" "image_segmenter" "custom" "custom" "custom")

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

  if is_gpu_instance;
  then
    if python scripts/validate_model_on_gpu.py; then
      echo "Model $model successfully loaded on GPU"
    else
      echo "Something went wrong, model $model did not load on GPU"
      exit 1
    fi
  fi

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
