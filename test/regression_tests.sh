#!/bin/bash

set -x
set -e

TS_REPO="https://github.com/pytorch/serve"
BRANCH=${1:-master}
ROOT_DIR="/workspace/"
CODEBUILD_WD=$(pwd)
MODEL_STORE=$ROOT_DIR"/model_store"
TS_LOG_FILE="/tmp/ts.log"
TEST_EXECUTION_LOG_FILE="/tmp/test_exec.log"


install_torchserve_from_source() {
  echo "Cloning & Building Torchserve Repo from " $1

  pip install  mock pytest pylint pytest-mock pytest-cov transformers
  sudo apt-get -y install nodejs-dev node-gyp libssl1.0-dev
  sudo apt-get -y install npm
  sudo npm install -g n
  sudo n latest
  export PATH="$PATH"
  sudo npm install -g newman newman-reporter-html

  # Clone & Build TorchServe
  git clone -b $2 $1
  cd serve
  echo "Installing torchserve torch-model-archiver from source"
  ./scripts/install_from_src_ubuntu
  echo "TS Branch : " $(git rev-parse --abbrev-ref HEAD) >> $3
  echo "TS Branch Commit Id : " $(git rev-parse HEAD) >> $3
  echo "Build date : " $(date) >> $3
  echo "Torchserve Succesfully installed"
}


generate_densenet_test_model_archive() {

  mkdir $1 && cd $1

  # Download & create DenseNet Model Archive
  wget https://download.pytorch.org/models/densenet161-8d451a50.pth
  torch-model-archiver --model-name densenet161_v1 \
	  --version 1.1 --model-file $ROOT_DIR/serve/examples/image_classifier/densenet_161/model.py \
	  --serialized-file $1/densenet161-8d451a50.pth \
	  --extra-files $ROOT_DIR/serve/examples/image_classifier/index_to_name.json \
	  --handler image_classifier
  rm densenet161-8d451a50.pth
  cd -

}


start_torchserve() {

  # Start Torchserve with Model Store
  torchserve --start --model-store $1 --models $1/densenet161_v1.mar &> $2
  sleep 10
  curl http://127.0.0.1:8081/models
  
}

start_secure_torchserve() {

  # Start Torchserve with Model Store
  torchserve --start --ts-config resources/config.properties --model-store $1 --models $1/densenet161_v1.mar &> $2
  sleep 10
  curl --insecure -X GET https://127.0.0.1:8444/models
}


stop_torch_serve() {
  torchserve --stop
}


delete_model_store_snapshots() {
  rm -f $MODEL_STORE/*
  rm -rf logs/
}


run_postman_test() {
  # Run Postman Scripts
  mkdir $ROOT_DIR/report/
  cd $CODEBUILD_WD/test/
  set +e
  # Run Management API Tests
  stop_torch_serve
  start_torchserve $MODEL_STORE $TS_LOG_FILE
  newman run -e postman/environment.json --bail --verbose postman/management_api_test_collection.json \
	  -r cli,html --reporter-html-export $ROOT_DIR/report/management_report.html >>$1 2>&1


  # Run Inference API Tests after Restart
  stop_torch_serve
  delete_model_store_snapshots
  start_torchserve $MODEL_STORE $TS_LOG_FILE
  newman run -e postman/environment.json --bail --verbose postman/inference_api_test_collection.json \
	  -d postman/inference_data.json -r cli,html --reporter-html-export $ROOT_DIR/report/inference_report.html >>$1 2>&1

  # Run Https test cases
  stop_torch_serve
  delete_model_store_snapshots
  start_secure_torchserve $MODEL_STORE $TS_LOG_FILE
  newman run --insecure -e postman/environment.json --bail --verbose postman/https_test_collection.json \
	  -r cli,html --reporter-html-export $ROOT_DIR/report/https_test_report.html >>$1 2>&1

  set -e
  cd -
}


run_pytest() {

  mkdir -p $ROOT_DIR/report/
  cd $CODEBUILD_WD/test/pytest
  stop_torch_serve
  set +e
  pytest . -v >>$1 2>&1
  set -e
  cd -

}

sudo rm -rf $ROOT_DIR && sudo mkdir $ROOT_DIR
sudo chown -R $USER:$USER $ROOT_DIR
cd $ROOT_DIR

echo "** Execuing TorchServe Regression Test Suite executon for " $TS_REPO " **"

install_torchserve_from_source $TS_REPO $BRANCH $TEST_EXECUTION_LOG_FILE
generate_densenet_test_model_archive $MODEL_STORE
run_postman_test $TEST_EXECUTION_LOG_FILE
run_pytest $TEST_EXECUTION_LOG_FILE

echo "** Tests Complete ** "
exit 0
