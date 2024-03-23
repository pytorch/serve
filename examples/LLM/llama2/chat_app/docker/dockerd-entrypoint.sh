#!/bin/bash
set -e

export LLAMA2_Q4_MODEL=/home/model-server/model-store/$MODEL_NAME/model/ggml-model-q4_0.gguf


create_model_cfg_yaml() {
  # Define the YAML content with a placeholder for the model name
  yaml_content="# TorchServe frontend parameters\nminWorkers: 1\nmaxWorkers: 1\nresponseTimeout: 1200\n#deviceType: \"gpu\"\n#deviceIds: [0,1]\n#torchrun:\n#    nproc-per-node: 1\n\nhandler:\n    model_name: \"${2}\"\n    manual_seed: 40"

  # Create the YAML file with the specified model name
  echo -e "$yaml_content" > "model-config-${1}.yaml"
}

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
    torch-model-archiver --model-name "$MODEL_NAME" --version 1.0 --handler llama_cpp_handler.py --config-file $MODEL_CFG -r requirements.txt --archive-format no-archive --export-path /home/model-server/model-store -f
    if [ -d "/home/model-server/model-store/model" ]; then
        mv /home/model-server/model-store/model /home/model-server/model-store/$MODEL_NAME/
    fi
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

quantize_model() {
    if [ ! -f "$LLAMA2_Q4_MODEL" ]; then
        tmp_model_name=$(echo "$MODEL_NAME" | sed 's/---/--/g')
        directory_path=/home/model-server/model-store/$MODEL_NAME/model/models--$tmp_model_name/snapshots/
        HF_MODEL_SNAPSHOT=$(find $directory_path -type d -mindepth 1)
        echo "Cleaning up previous build of llama-cpp"
        git clone https://github.com/ggerganov/llama.cpp.git build
        cd build
        make
        python -m pip install -r requirements.txt

        echo "Convert the 7B model to ggml FP16 format"
        python convert.py $HF_MODEL_SNAPSHOT --outfile ggml-model-f16.gguf

        echo "Quantize the model to 4-bits (using q4_0 method)"
        ./quantize ggml-model-f16.gguf $LLAMA2_Q4_MODEL q4_0

        cd ..
        echo "Saved quantized model weights to $LLAMA2_Q4_MODEL"
    fi
}

HF_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's/---/\//g')
if [[ "$1" = "serve" ]]; then
    shift 1
    create_model_cfg_yaml $MODEL_NAME $HF_MODEL_NAME
    create_model_archive $MODEL_NAME "model-config-$MODEL_NAME.yaml"
    download_model $MODEL_NAME $HF_MODEL_NAME
    quantize_model
    streamlit run torchserve_server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
