#!/bin/bash
HF_REPO_URL = ""
MODEL_NAME = ""
for arg in "$@"
do
    case $arg in
        -h|--help)
            echo "options:"
            echo "-t, --task specify task for the pipeline supported by HF Hub model"
            echo "-n, --model-name specify name for locally saved model repo"
            echo "-f, --framework specify the framework for the loaded model available in the repo, should be either \"pt\" or \"tf\""
            echo "-u, --hf-hub-link specify the link to the repo available in HF model hub"
            exit 0
            ;;
        -t|--task)
            sed -i "s/task=\".*/task=\"$2\"/g" scripts/torchserve_vitxxsmall_handler.py
            shift 2
            ;;
        -n|--model-name)
            MODEL_NAME="$2"
            echo "MODEL_NAME is $MODEL_NAME"
            shift 2
            ;;
        -f|--framework)
            sed -i "s/framework=\".*/framework=\"$2\"/g" scripts/torchserve_vitxxsmall_handler.py
            shift 2
            ;;
        -u|--hf-hub-link)
            HF_REPO_URL="$2"
            git lfs install
            git clone $HF_REPO_URL HF-models/$MODEL_NAME/
            cd HF-models/$MODEL_NAME/
            git lfs install
            git lfs pull
            cd ../..
            touch dummy_file.pth
            torch-model-archiver --model-name $MODEL_NAME --serialized-file dummy_file.pth --version 1.0 --handler scripts/torchserve_vitxxsmall_handler.py --export-path model-store -r requirements.txt
            rm -f dummy_file.pth
            shift 2
            ;;
    esac
done