#!/bin/bash
set -e

MODEL_DIR=$(echo "$MODEL_NAME" | sed 's/\//---/g')

export LLAMA_Q4_MODEL=/home/model-server/model-store/$MODEL_DIR/model/ggml-model-q4_0.gguf


create_model_cfg_yaml() {
  # Define the YAML content with a placeholder for the model name
  yaml_content="# TorchServe frontend parameters\nminWorkers: 1\nmaxWorkers: 1\nresponseTimeout: 1200\n#deviceType: \"gpu\"\n#deviceIds: [0,1]\n#torchrun:\n#    nproc-per-node: 1\n\nhandler:\n    model_name: \"${2}\"\n    manual_seed: 40"

  # Create the YAML file
  echo -e "$yaml_content" > "model-config.yaml"
}

create_model_archive() {
    MODEL_DIR=$1
    echo "Create model archive for ${MODEL_DIR} if it doesn't already exist"
    if [ -d "/home/model-server/model-store/$MODEL_DIR" ]; then
        echo "Model archive for $MODEL_DIR exists."
    fi
    if [ -d "/home/model-server/model-store/$MODEL_DIR/model" ]; then
        echo "Model already download"
        mv /home/model-server/model-store/$MODEL_DIR/model /home/model-server/model-store/
    else
        echo "Model needs to be downloaded"
    fi
    torch-model-archiver --model-name "$MODEL_DIR" --version 1.0 --handler llama_cpp_handler.py --config-file "model-config.yaml" -r requirements.txt --archive-format no-archive --export-path /home/model-server/model-store -f
    if [ -d "/home/model-server/model-store/model" ]; then
        mv /home/model-server/model-store/model /home/model-server/model-store/$MODEL_DIR/
    fi
}

download_model() {
   MODEL_DIR=$1
   MODEL_NAME=$2
    if [ -d "/home/model-server/model-store/$MODEL_DIR/model" ]; then
        echo "Model $MODEL_NAME already downloaded"
    else
        echo "Downloading  model $MODEL_NAME"
        python Download_model.py --model_path /home/model-server/model-store/$MODEL_DIR/model --model_name $MODEL_NAME
    fi
}

quantize_model() {
    if [ ! -f "$LLAMA_Q4_MODEL" ]; then
        tmp_model_name=$(echo "$MODEL_DIR" | sed 's/---/--/g')
        directory_path=/home/model-server/model-store/$MODEL_DIR/model/models--$tmp_model_name/snapshots/
        HF_MODEL_SNAPSHOT=$(find $directory_path -type d -mindepth 1)
        cd build

        echo "Convert the model to ggml FP16 format"
        if [[ $MODEL_NAME == *"Meta-Llama-3"* ]]; then
            python convert.py $HF_MODEL_SNAPSHOT --vocab-type bpe,hfft --outfile ggml-model-f16.gguf
        else
            python convert.py $HF_MODEL_SNAPSHOT --outfile ggml-model-f16.gguf
        fi

        echo "Quantize the model to 4-bits (using q4_0 method)"
        ./quantize ggml-model-f16.gguf $LLAMA_Q4_MODEL q4_0

        cd ..
        echo "Saved quantized model weights to $LLAMA_Q4_MODEL"
    fi
}

if [[ "$1" = "serve" ]]; then
    shift 1
    create_model_cfg_yaml $MODEL_DIR $MODEL_NAME
    create_model_archive $MODEL_DIR
    download_model $MODEL_DIR $MODEL_NAME
    quantize_model
    streamlit run torchserve_server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
