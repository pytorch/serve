#!/bin/bash
set -e

MODEL_DIR_LLM=$(echo "$MODEL_NAME_LLM" | sed 's/\//---/g')
export LLAMA_Q4_MODEL=/home/model-server/model-store/$MODEL_DIR_LLM/model/ggml-model-q4_0.gguf

MODEL_DIR_SD=$(echo "$MODEL_NAME_SD" | sed 's/\//---/g')

create_model_cfg_yaml() {
  # Define the YAML content with a placeholder for the model name
  yaml_content="# TorchServe frontend parameters\nminWorkers: 1\nmaxWorkers: 1\nresponseTimeout: 1200\n#deviceType: \"gpu\"\n#deviceIds: [0,1]\n#torchrun:\n#    nproc-per-node: 1\n\nhandler:\n    model_name: \"${2}\"\n    manual_seed: 40"

  # Create the YAML file
  echo -e "$yaml_content" > "model-config.yaml"
}

create_model_archive() {
    MODEL_DIR_LLM=$1
    echo "Create model archive for ${MODEL_DIR_LLM} if it doesn't already exist"
    if [ -d "/home/model-server/model-store/$MODEL_DIR_LLM" ]; then
        echo "Model archive for $MODEL_DIR_LLM exists."
    fi

    if [ -d "/home/model-server/model-store/$MODEL_DIR_LLM/model" ]; then
        echo "$MODEL_DIR_LLM Model already downloaded"
        mv /home/model-server/model-store/$MODEL_DIR_LLM/model /home/model-server/model-store/
    else
        echo "Model needs to be downloaded"
    fi

    torch-model-archiver --model-name "$MODEL_DIR_LLM" --version 1.0 \
    --handler llama_cpp_handler.py --config-file "model-config.yaml" \
    -r requirements.txt --archive-format no-archive \
    --export-path /home/model-server/model-store -f

    if [ -d "/home/model-server/model-store/model" ]; then
        mv /home/model-server/model-store/model /home/model-server/model-store/$MODEL_DIR_LLM/
    fi
}

download_model() {
   MODEL_DIR_LLM=$1
   MODEL_NAME_LLM=$2
    if [ -d "/home/model-server/model-store/$MODEL_DIR_LLM/model" ]; then
        echo "Model $MODEL_NAME_LLM already downloaded"
    else
        echo "Downloading  model $MODEL_NAME_LLM"
        python Download_model.py \
        --model_path /home/model-server/model-store/$MODEL_DIR_LLM/model \
        --model_name $MODEL_NAME_LLM
    fi
}

quantize_model() {
    if [ ! -f "$LLAMA_Q4_MODEL" ]; then
        tmp_model_name=$(echo "$MODEL_DIR_LLM" | sed 's/---/--/g')
        directory_path=/home/model-server/model-store/$MODEL_DIR_LLM/model/models--$tmp_model_name/snapshots/
        HF_MODEL_SNAPSHOT=$(find $directory_path -mindepth 1 -type d)

        cd build

        echo "Convert the model to ggml FP16 format"
        python convert_hf_to_gguf.py $HF_MODEL_SNAPSHOT --outfile ggml-model-f16.gguf
        
        echo "Quantize the model to 4-bits (using q4_0 method)"
        ./llama-quantize ggml-model-f16.gguf $LLAMA_Q4_MODEL q4_0

        cd ..
        echo "Saved quantized model weights to $LLAMA_Q4_MODEL"
    else
        echo "Quantized model available at $LLAMA_Q4_MODEL"
    fi
}

setup_sd() {
    
    echo -e "\nPreparing $MODEL_NAME_SD"
    
    pushd sd

    if [ -d "/home/model-server/model-store/$MODEL_DIR_SD/model" ]; then
        echo "Model $MODEL_NAME_SD already downloaded"
    else
        echo "Downloading  model $MODEL_NAME_SD"
        python Download_model.py --model_path /home/model-server/model-store/$MODEL_DIR_SD/model --model_name $MODEL_NAME_SD
    fi

    echo "Create model archive for ${MODEL_DIR_SD} if it doesn't already exist"
    if [ -d "/home/model-server/model-store/$MODEL_DIR_SD/MAR-INF" ]; then
        echo "Model archive for $MODEL_DIR_SD exists."
    else
        # Create model archive
        torch-model-archiver --model-name "$MODEL_DIR_SD" --version 1.0 \
        --handler stable_diffusion_handler.py --config-file model-config.yaml \
        --extra-files "./pipeline_utils.py" --archive-format no-archive

        mv /home/model-server/model-store/$MODEL_DIR_SD/model $MODEL_DIR_SD
        mv $MODEL_DIR_SD /home/model-server/model-store/

        echo "Model archive created at /home/model-server/model-store/$MODEL_DIR_SD/"
    fi

    popd
}


if [[ "$1" = "serve" ]]; then
    shift 1
    create_model_cfg_yaml $MODEL_DIR_LLM $MODEL_NAME_LLM
    create_model_archive $MODEL_DIR_LLM
    download_model $MODEL_DIR_LLM $MODEL_NAME_LLM
    quantize_model
    setup_sd
    streamlit run torchserve_server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
