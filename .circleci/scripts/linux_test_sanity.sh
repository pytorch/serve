#!/bin/bash

source ts_scripts/install_utils

MODELS=("fastrcnn" "fcn_resnet_101" "deeplabv3_resnet_101_eager" "my_text_classifier_v4" "resnet-18" "my_text_classifier_scripted_v3" "alexnet_scripted" "fcn_resnet_101_scripted"
           "deeplabv3_resnet_101_scripted" "distill_bert_qa_eager" "bert_token_classification_no_torchscript" "bert_seqc_without_torchscript")
MODEL_INPUTS=("examples/object_detector/persons.jpg,docs/images/blank_image.jpg" "examples/image_segmenter/persons.jpg" "examples/image_segmenter/persons.jpg"
 "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg" "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg"
 "examples/image_segmenter/persons.jpg" "examples/image_segmenter/persons.jpg" "examples/Huggingface_Transformers/QA_artifacts/sample_text.txt" 

 "examples/Huggingface_Transformers/Token_classification_artifacts/sample_text.txt" "examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt")
HANDLERS=("object_detector" "image_segmenter" "image_segmenter" "text_classification" "image_classifier" "text_classification" "image_classifier" "image_segmenter" "image_segmenter" "custom" "custom" "custom")

mkdir model_store

start_torchserve

for i in ${!MODELS[@]};
do
  model=${MODELS[$i]}
  inputs=$(echo ${MODEL_INPUTS[$i]} | tr "," "\n")
  handler=${HANDLERS[$i]}
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

run_markdown_link_checker
