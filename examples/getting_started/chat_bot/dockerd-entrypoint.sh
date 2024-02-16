#!/bin/bash
set -e

MODEL_NAME="llama2-7b-chat"

if [[ "$1" = "serve" ]]; then
    shift 1
    echo "Create model archive if it doesn't already exist"
    if [ -d "/home/model-server/model-store/$MODEL_NAME" ]; then
        echo "Model archive exists."
    else
        torch-model-archiver --model-name $MODEL_NAME --version 1.0 --handler custom_handler.py --config-file model-config.yaml -r requirements.txt --archive-format no-archive --export-path /home/model-server/model-store || echo "Model archive already exists"
    fi
    echo "Download model if not already done"
    if [ -d "/home/model-server/model-store/$MODEL_NAME/model" ]; then
        echo "Model already downloaded"
    else
        python Download_model.py --model_path /home/model-server/model-store/$MODEL_NAME/model --model_name meta-llama/Llama-2-7b-chat-hf
    fi

    streamlit run torchserve_server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
