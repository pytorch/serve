#!/bin/bash
set -e

# Replace "/" in HF model name with "---". Ex. meta-llama/Llama-3.2-1B -> meta-llama---Llama-3.2-1B
MODEL_DIR_LLM=$(echo "$MODEL_NAME_LLM" | sed 's/\//---/g')
MODEL_DIR_SD=$(echo "$MODEL_NAME_SD" | sed 's/\//---/g')

setup_llm() {

    echo -e "\nPreparing $MODEL_NAME_LLM"

    MODEL_DIR_LLM=$1
    MODEL_NAME_LLM=$2

    pushd llm
    if [ -d "/home/model-server/model-store/$MODEL_DIR_LLM/model" ]; then
        echo "Model $MODEL_DIR_LLM already downloaded."
    else
        echo "Downloading  model $MODEL_NAME_LLM"
        python download_model_llm.py --model_path /home/model-server/model-store/$MODEL_DIR_LLM/model --model_name $MODEL_NAME_LLM
    fi


    if [ -d "/home/model-server/model-store/$MODEL_DIR_LLM/MAR-INF" ]; then
        echo "Model archive for $MODEL_DIR_LLM exists."
    else
        echo "Creating model archive for ${MODEL_DIR_LLM} ..."
        # Create model archive
        torch-model-archiver \
            --model-name "$MODEL_DIR_LLM" \
            --version 1.0 \
            --handler llm_handler.py \
            --config-file model-config.yaml \
            --archive-format no-archive

        mv /home/model-server/model-store/$MODEL_DIR_LLM/model $MODEL_DIR_LLM
        mv $MODEL_DIR_LLM /home/model-server/model-store

        echo "Model archive created at /home/model-server/model-store/$MODEL_DIR_LLM/"
    fi

    popd
}

setup_sd() {
    
    echo -e "\nPreparing $MODEL_NAME_SD"
    
    pushd sd

    if [ -d "/home/model-server/model-store/$MODEL_DIR_SD/model" ]; then
        echo "Model $MODEL_NAME_SD already downloaded"
    else
        echo "Downloading  model $MODEL_NAME_SD"
        python download_model_sd.py --model_path /home/model-server/model-store/$MODEL_DIR_SD/model --model_name $MODEL_NAME_SD
    fi

    if [ -d "/home/model-server/model-store/$MODEL_DIR_SD/MAR-INF" ]; then
        echo "Model archive for $MODEL_DIR_SD exists."
    else
        echo "Creating model archive for ${MODEL_DIR_SD} ..."
                # Create model archive
        torch-model-archiver \
            --model-name "$MODEL_DIR_SD" \
            --version 1.0 \
            --handler stable_diffusion_handler.py \
            --config-file model-config.yaml \
            --archive-format no-archive

        mv /home/model-server/model-store/$MODEL_DIR_SD/model $MODEL_DIR_SD
        mv $MODEL_DIR_SD /home/model-server/model-store/

        echo "Model archive created at /home/model-server/model-store/$MODEL_DIR_SD/"
    fi

    popd
}


if [[ "$1" = "serve" ]]; then
    shift 1
    setup_llm $MODEL_DIR_LLM $MODEL_NAME_LLM
    setup_sd
    streamlit run server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
