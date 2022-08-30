#!/bin/bash
TASK=""
OUTFILE=""
#TODO: Add tasks supported by `Transformer_handler_generalized.py` to this list progressively
SUPPORTED_TASKS=("image-classification" "sentiment-analysis")
for arg in "$@"
do
    case $arg in
        -h|--help)
            echo "options:"
            echo "-t, --task specify task for which the HF handler is to be created"
            echo "-o, --output-file specify the absolute path to the handler file to be created"
            exit 0
            ;;
        -t|--task)
            TASK="$2"
            #source: https://unix.stackexchange.com/questions/111508/bash-test-if-word-is-in-set
            if ! (echo "${SUPPORTED_TASKS[@]}" | fgrep -wq "$TASK") ; then
                echo "Task not supported yet"
                exit 1
            fi
            shift 2
            ;;
        -o|--output-file)
            mkdir -p "$(dirname "$2")" && touch "$2"
            if ! [[ $2 == *.py ]] ; then
                echo "Enter a valid output path that ends with `.py`"
                exit 1
            fi
            #Creates the handler specific to the task
            python Transformer_handler_generalized.py --task $TASK --output-file $2
            shift 2
            ;;