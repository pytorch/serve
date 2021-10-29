#!/bin/bash

BASE_DIR=$(pwd)
MODEL_STORE="/workspace/model_store"

generate_densenet_test_model_archive() {
  mkdir -p $1 && cd $1
  local MODEL_FILE_NAME="densenet161-8d451a50.pth"
  # Download & create DenseNet Model Archive
  curl https://download.pytorch.org/models/$MODEL_FILE_NAME -o $MODEL_FILE_NAME
  torch-model-archiver --model-name densenet161_v1 \
	  --version 1.1 --model-file $BASE_DIR/examples/image_classifier/densenet_161/model.py \
	  --serialized-file $1/$MODEL_FILE_NAME \
	  --extra-files $BASE_DIR/examples/image_classifier/index_to_name.json \
	  --handler image_classifier
  rm $MODEL_FILE_NAME
}

run_pytest() {
  cd $BASE_DIR/test/pytest
  python -m pytest -v ./
}

generate_densenet_test_model_archive $MODEL_STORE
run_pytest