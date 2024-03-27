#!/bin/bash
set -e

echo "Model Name: $MODEL_NAME"
echo "Torch Compile: $TORCH_COMPILE"
create_model_cfg_yaml() {
  # Define the YAML content with a placeholder for the model name
  yaml_content="# TorchServe frontend parameters\nminWorkers: 1\nmaxWorkers: 1\nresponseTimeout: 1200\nhandler:\n profile: true"

  # Create the YAML file with the specified model name
  echo -e "$yaml_content" > "model-config-${1}.yaml"

  # Append the backend configuration if TorchScript compilation is enabled
  if [ "$TORCH_COMPILE" = true ]; then
    echo "pt2 : {backend: inductor}" >> "model-config-${1}.yaml"
  fi
}

get_model_weights_url() {
  local model_name="$1"
  local json_file="/home/model-server/getting_started/index_mapping/torchvision_models.json"
  local url
  url=$(jq -r "to_entries[] | select(.key == \"$model_name\") | .value" "$json_file")
  if [ -z "$url" ]; then
    echo "Error: Model name not found in JSON file" >&2
    exit 1
  fi
  echo "$url"
}
create_model_archive() {
    MODEL_NAME=$1
    if [ ! -d "/home/model-server/model-store/$MODEL_NAME" ]; then
        mkdir -p "/home/model-server/model-store/$MODEL_NAME"
    fi
    if [[ $MODEL_NAME == "bert"* ]]; then
        download_model $MODEL_NAME
        cp configs/$MODEL_NAME.json setup_config.json
        cp index_mapping/${MODEL_NAME}_index_to_name.json index_to_name.json
        torch-model-archiver --model-name $MODEL_NAME --version 1.0 --serialized-file /home/model-server/model-store/$MODEL_NAME/model/model.safetensors --handler ./Transformer_handler_generalized.py --config-file model-config-$MODEL_NAME.yaml --archive-format no-archive --extra-files "/home/model-server/model-store/$MODEL_NAME/model/config.json,./setup_config.json,./index_to_name.json" --export-path /home/model-server/model-store -f
    else
        echo "Get model weights"
        local url=$(get_model_weights_url $MODEL_NAME)
        local filename=$(basename "$url")
        echo "url $url"
        if [ ! -f /home/model-server/model-store/$MODEL_NAME/$filename ]; then
            wget $url -O /home/model-server/model-store/$MODEL_NAME/$filename
        fi
        echo "Create model archive for ${MODEL_NAME}"
        local handler=image_classifier
        if [[ $MODEL_NAME == "fasterrcnn" ]]; then
            handler=object_detector
        fi
        cp index_mapping/$handler.json index_to_name.json
        torch-model-archiver --model-name "$MODEL_NAME" --version 1.0 --model-file models/$MODEL_NAME.py --serialized-file /home/model-server/model-store/$MODEL_NAME/$filename  --handler $handler --config-file model-config-$MODEL_NAME.yaml --archive-format no-archive --extra-files index_to_name.json --export-path /home/model-server/model-store -f
    fi

}

download_model() {
   MODEL_NAME=$1
    if [ -d "/home/model-server/model-store/$MODEL_NAME/model" ]; then
        echo "Model $MODEL_NAME already downloaded"
    else
        echo "Downloading  model $MODEL_NAME"
        python Download_Transformer_models.py --model_path /home/model-server/model-store/$MODEL_NAME/model --cfg $MODEL_NAME.json
    fi
}

if [[ "$1" = "serve" ]]; then
    shift 1

    create_model_cfg_yaml $MODEL_NAME
    create_model_archive $MODEL_NAME
    torchserve --start --ncs --models $MODEL_NAME=$MODEL_NAME --ts-config /home/model-server/config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
