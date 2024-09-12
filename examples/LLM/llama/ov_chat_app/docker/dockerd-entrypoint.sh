#!/bin/bash
set -e

MODEL_DIR_LLM=$(echo "$MODEL_NAME_LLM" | sed 's/\//---/g')
MODEL_DIR_SD=$(echo "$MODEL_NAME_SD" | sed 's/\//---/g')


create_model_cfg_yaml() {
  # Define the YAML content with a placeholder for the model name
  yaml_content="# TorchServe frontend parameters\nminWorkers: 1\nmaxWorkers: 1\nresponseTimeout: 1200\n#deviceType: \"gpu\"\n#deviceIds: [0,1]\n#torchrun:\n#    nproc-per-node: 1\n\nhandler:\n    model_name: \"${2}\"\n    manual_seed: 40"

  # Create the YAML file
  echo -e "$yaml_content" > "model-config.yaml"
}


setup_llm() {

    echo -e "\nPreparing $MODEL_NAME_LLM"

    pushd gpt-fast

    MODEL_DIR_LLM=$1
    MODEL_NAME_LLM=$2
    if [ -f "/home/model-server/model-store/$MODEL_DIR_LLM/checkpoints/$MODEL_NAME_LLM/model_int4.g32.pth" ]; then
        echo "Model $MODEL_NAME_LLM already downloaded."
    else
        echo "Downloading  model $MODEL_NAME_LLM"
        ./scripts/prepare.sh "$MODEL_NAME_LLM"
        python quantize.py --checkpoint_path /home/model-server/chat_bot/gpt-fast/checkpoints/$MODEL_NAME_LLM/model.pth --mode int4 --groupsize 32
    fi

    popd

    pushd llm

    if [ -d "/home/model-server/model-store/$MODEL_DIR_LLM/MAR-INF" ]; then
        echo "Model archive for $MODEL_DIR_LLM exists."
    else
        echo "Creating model archive for ${MODEL_DIR_LLM} ..."
        # Create model archive
        torch-model-archiver --model-name "$MODEL_DIR_LLM" --version 1.0 \
        --handler llama_handler.py --config-file model-config.yaml \
        --extra-files "/home/model-server/chat_bot/gpt-fast/generate.py,/home/model-server/chat_bot/gpt-fast/model.py,/home/model-server/chat_bot/gpt-fast/quantize.py,/home/model-server/chat_bot/gpt-fast/tp.py,/home/model-server/chat_bot/gpt-fast/tokenizer.py" \
        --archive-format no-archive

        mv /home/model-server/chat_bot/gpt-fast/checkpoints $MODEL_DIR_LLM
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
        python Download_model.py --model_path /home/model-server/model-store/$MODEL_DIR_SD/model --model_name $MODEL_NAME_SD
    fi

    if [ -d "/home/model-server/model-store/$MODEL_DIR_SD/MAR-INF" ]; then
        echo "Model archive for $MODEL_DIR_SD exists."
    else
        echo "Creating model archive for ${MODEL_DIR_SD} ..."
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
    setup_llm $MODEL_DIR_LLM $MODEL_NAME_LLM
    setup_sd
    streamlit run torchserve_server_app.py --server.port 8084 &
    streamlit run client_app.py --server.port 8085
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
