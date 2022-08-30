#!/bin/bash
HF_REPO_URL=""
MODEL_NAME=""
HANDLER_PATH=""
REQ=""
WORKDIR=""
#TODO: Add condition to give error if $HANDLER_PATH in `prepare_mar_from_hf.sh` does not end with .py
for arg in "$@"
do
    case $arg in
        -h|--help)
            echo "options:"
            echo "-p, --handler-path Path to the handler .py file for creating model archive"
            echo "-t, --task specify task for the pipeline supported by HF Hub model"
            echo "-n, --model-name specify name for locally saved model repo"
            echo "-f, --framework specify the framework for the loaded model available in the repo, should be either \"pt\" or \"tf\""
            echo "-r, --requirements Path to the requirements.txt file, it MUST include `huggingface` as a dependency"
            echo "-d, --workdir Path to the directory that contains the `model-store` directory"
            echo "-u, --hf-hub-link specify the link to the repo available in HF model hub"
            exit 0
            ;;
        -p|--handler-path)
            HANDLER_PATH="$2"
            if ! [[ $HANDLER_PATH == *.py ]] ; then
                echo "Enter a valid output path that ends with `.py`"
                exit 1
            fi
            shift 2
            ;;
        -t|--task)
            sed -i "s/task=\".*/task=\"$2\"/g" $HANDLER_PATH
            shift 2
            ;;
        -n|--model-name)
            MODEL_NAME="$2"
            echo "MODEL_NAME is $MODEL_NAME"
            shift 2
            ;;
        -f|--framework)
            sed -i "s/framework=\".*/framework=\"$2\"/g" $HANDLER_PATH
            shift 2
            ;;
        -r|--requirements)
            REQ="$2"
            shift 2
            ;;
        -d|--workdir)
            WORKDIR="$2"
            shift 2
            ;;
        -u|--hf-hub-link)
            HF_REPO_URL="$2"
            git lfs install
            git clone $HF_REPO_URL HF-models/$MODEL_NAME/
            cd HF-models/$MODEL_NAME/ && git lfs install && git lfs pull && cd ../..
            touch dummy_file.pth
            torch-model-archiver --model-name $MODEL_NAME --serialized-file dummy_file.pth --version 1.0 --handler $HANDLER_PATH --export-path $WORKDIR/model-store -r $REQ
            rm -f dummy_file.pth
            shift 2
            ;;
    esac
done
