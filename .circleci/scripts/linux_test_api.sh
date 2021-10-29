#!/bin/bash

BASE_DIR="test"
MODEL_STORE_DIR="model_store"

ARTIFACTS_BASE_DIR="artifacts"
ARTIFACTS_MANAGEMENT_DIR="$ARTIFACTS_BASE_DIR/management"
ARTIFACTS_INFERENCE_DIR="$ARTIFACTS_BASE_DIR/inference"
ARTIFACTS_HTTPS_DIR="$ARTIFACTS_BASE_DIR/https"

TS_CONSOLE_LOG_FILE="ts_console.log"
TS_CONFIG_FILE_HTTPS="resources/config.properties"

POSTMAN_ENV_FILE="postman/environment.json"
POSTMAN_COLLECTION_MANAGEMENT="postman/management_api_test_collection.json"
POSTMAN_COLLECTION_INFERENCE="postman/inference_api_test_collection.json"
POSTMAN_COLLECTION_HTTPS="postman/https_test_collection.json"
POSTMAN_DATA_FILE_INFERENCE="postman/inference_data.json"

REPORT_FILE="report.html"

start_ts() {
  torchserve --ncs --start --model-store $1 >> $2 2>&1
  sleep 10
}

start_ts_secure() {
  torchserve --ncs --start --ts-config $TS_CONFIG_FILE_HTTPS --model-store $1 >> $2 2>&1
  sleep 10
}

stop_ts() {
  torchserve --stop
  sleep 10
}

cleanup_model_store(){
  rm -rf $MODEL_STORE_DIR/*
}

move_logs(){
  mv $1 logs/
  mv logs/ $2
}

trigger_management_tests(){
  start_ts $MODEL_STORE_DIR $TS_CONSOLE_LOG_FILE
  newman run -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_MANAGEMENT \
             -r cli,html --reporter-html-export $ARTIFACTS_MANAGEMENT_DIR/$REPORT_FILE --verbose
  local EXIT_CODE=$?
  stop_ts
  move_logs $TS_CONSOLE_LOG_FILE $ARTIFACTS_MANAGEMENT_DIR
  cleanup_model_store
  return $EXIT_CODE
}

trigger_inference_tests(){
  start_ts $MODEL_STORE_DIR $TS_CONSOLE_LOG_FILE
  newman run -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_INFERENCE -d $POSTMAN_DATA_FILE_INFERENCE \
             -r cli,html --reporter-html-export $ARTIFACTS_INFERENCE_DIR/$REPORT_FILE --verbose
  local EXIT_CODE=$?
  stop_ts
  move_logs $TS_CONSOLE_LOG_FILE $ARTIFACTS_INFERENCE_DIR
  cleanup_model_store
  return $EXIT_CODE
}

trigger_https_tests(){
  start_ts_secure $MODEL_STORE_DIR $TS_CONSOLE_LOG_FILE
  newman run --insecure -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_HTTPS \
             -r cli,html --reporter-html-export $ARTIFACTS_HTTPS_DIR/$REPORT_FILE --verbose
  local EXIT_CODE=$?
  stop_ts
  move_logs $TS_CONSOLE_LOG_FILE $ARTIFACTS_HTTPS_DIR
  cleanup_model_store
  return $EXIT_CODE
}

cd $BASE_DIR
mkdir -p $MODEL_STORE_DIR $ARTIFACTS_MANAGEMENT_DIR $ARTIFACTS_INFERENCE_DIR $ARTIFACTS_HTTPS_DIR

case $1 in
   'management')
      trigger_management_tests
      exit $?
      ;;
   'inference')
      trigger_inference_tests
      exit $?
      ;;
   'https')
      trigger_https_tests
      exit $?
      ;;
   'ALL')
      trigger_management_tests
      MGMT_EXIT_CODE=$?
      trigger_inference_tests
      INFR_EXIT_CODE=$?
      trigger_https_tests
      HTTPS_EXIT_CODE=$?
      # If any one of the tests fail, exit with error
      if [ "$MGMT_EXIT_CODE" -ne 0 ] || [ "$INFR_EXIT_CODE" -ne 0 ] || [ "$HTTPS_EXIT_CODE" -ne 0 ]
      then exit 1
      fi
      ;;
   *)
     echo $1 'Invalid'
     echo 'Please specify any one of - management | inference | https | ALL'
     exit 1
     ;;
esac