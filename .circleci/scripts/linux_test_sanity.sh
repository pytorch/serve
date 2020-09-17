#!/bin/bash

source scripts/install_utils

MODELS=("fastrcnn" "fcn_resnet_101" "my_text_classifier_v2" "resnet-18" "my_text_classifier_scripted_v2" "alexnet_scripted" "fcn_resnet_101_scripted"
           "roberta_qa_no_torchscript" "bert_token_classification_no_torchscript" "bert_seqc_without_torchscript")
MODEL_INPUTS=("examples/object_detector/persons.jpg,docs/images/blank_image.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg"
 "examples/text_classification/sample_text.txt" "examples/image_classifier/kitten.jpg" "examples/image_segmenter/fcn/persons.jpg" "examples/Huggingface_Transformers/QA_artifacts/sample_text.txt"
 "examples/Huggingface_Transformers/Token_classification_artifacts/sample_text.txt" "examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt")
HANDLERS=("object_detector" "image_segmenter" "text_classification" "image_classifier" "text_classification" "image_classifier" "image_segmenter" "custom" "custom" "custom")

torch_api_types=("cpp" "python")
#Here "both" means both cpp and python API are supported by handler
torch_api_support=("python" "python" "python" "python" "python" "both" "python" "python" "python" "python")

mkdir model_store

start_torchserve

for i in ${!models[@]};
do
  for api_type_idx in ${!torch_api_types[@]};
  do
    api_type=${torch_api_types[$api_type_idx]}
    supported_api=${torch_api_support[$i]}
    if [ "$supported_api" == "python" ] &&  [ "$api_type" == "cpp" ]; then
      continue;
    fi
    model=${models[$i]}
    inputs=$(echo ${model_inputs[$i]} | tr "," "\n")
    handler=${handlers[$i]}
    register_model "$model" "$api_type"
    for input in ${inputs[@]};
    do
      run_inference "$model" "$input"
    done

    #skip unregistering resnet-18 model to test snapshot feature with restart
    if [ "$model" != "resnet-18" ]
    then
      unregister_model "$model"
    fi
    echo "$handler default handler with ${api_type} API is stable."
    done
done


stop_torchserve

# restarting torchserve
# this should restart with the generated snapshot and resnet-18 model should be automatically registered

start_torchserve

run_inference resnet-18 examples/image_classifier/kitten.jpg

stop_torchserve

run_markdown_link_checker