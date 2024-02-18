#!/bin/bash
set -e

create_model_archive() {
    MODEL_NAME=$1
    MODEL_CFG=$2
    echo "Create model archive for ${MODEL_NAME} if it doesn't already exist"
    if [ -d "/home/model-server/model-store/$MODEL_NAME" ]; then
        echo "Model archive for $MODEL_NAME exists."
    fi
    if [ -d "/home/model-server/model-store/$MODEL_NAME/model" ]; then
        echo "Model already download"
        mv /home/model-server/model-store/$MODEL_NAME/model /home/model-server/model-store/
    else
        echo "Model needs to be downloaded"
    fi
    torch-model-archiver --model-name "$MODEL_NAME" --version 1.0 --handler custom_handler.py --config-file $MODEL_CFG -r requirements.txt --archive-format no-archive --export-path /home/model-server/model-store -f
    mv /home/model-server/model-store/model /home/model-server/model-store/$MODEL_NAME/
}

download_model() {
   MODEL_NAME=$1
   HF_MODEL_NAME=$2
    if [ -d "/home/model-server/model-store/$MODEL_NAME/model" ]; then
        echo "Model $HF_MODEL_NAME already downloaded"
    else
        echo "Downloading  model $HF_MODEL_NAME"
        python Download_model.py --model_path /home/model-server/model-store/$MODEL_NAME/model --model_name $HF_MODEL_NAME
    fi

}

if [[ "$1" = "serve" ]]; then
    shift 1
    create_model_archive $MODEL_NAME_1 "model-config-$MODEL_NAME_1.yaml"
    download_model $MODEL_NAME_1 "mistralai/Mistral-7B-v0.1"
    create_model_archive $MODEL_NAME_2 "model-config-$MODEL_NAME_2.yaml"
    download_model $MODEL_NAME_2 "meta-llama/Llama-2-7b-chat-hf"
    streamlit run torchserve_server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
