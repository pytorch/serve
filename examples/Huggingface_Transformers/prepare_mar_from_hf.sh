#!/bin/bash
HF_REPO_URL=""
MODEL_NAME=""
HANDLER_PATH=""
WORKDIR=""
for arg in "$@"
do
    case $arg in
        -h|--help)
            echo "options:"
            echo "-p, --handler-path Path to the handler .py file for creating model archive"
            echo "-n, --model-name specify name for locally saved model repo"
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
        -n|--model-name)
            MODEL_NAME="$2"
            echo "MODEL_NAME is $MODEL_NAME"
            shift 2
            ;;
        -d|--workdir)
            WORKDIR="$2"
            shift 2
            ;;
        -u|--hf-hub-link)
            HF_REPO_URL="$2"
            git lfs install
            git clone $HF_REPO_URL $WORKDIR/HF-models/$MODEL_NAME/
            cd $WORKDIR/HF-models/$MODEL_NAME/ && git lfs install && git lfs pull && cd ../..
            touch dummy_file.pth
            echo "transformers==4.21.2" > transformers_req.txt
            torch-model-archiver --model-name $MODEL_NAME --serialized-file dummy_file.pth --version 1.0 --handler $HANDLER_PATH --export-path $WORKDIR/model-store -r transformers_req.txt
            cd $WORKDIR
            rm -f dummy_file.pth
            shift 2
            ;;
    esac
done
