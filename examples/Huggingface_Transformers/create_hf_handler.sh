#!/bin/bash
TASK=""
OUTFILE=""
#TODO: Add tasks supported by `Transformer_handler_generalized.py` to this list progressively
SUPPORTED_TASKS=("image-classification" "sentiment-analysis")
SUPPORTED_FRAMEWORKS=("pt" "tf")
for arg in "$@"
do
    case $arg in
        -h|--help)
            echo "options:"
            echo "-t, --task specify task for which the HF handler is to be created"
            echo "-f, --framework specify the framework for the loaded model available in the repo, should be either \"pt\" or \"tf\""
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
        -f|--framework)
            FRAMEWORK="$2"
            if ! (echo "${SUPPORTED_FRAMEWORKS[@]}" | fgrep -wq "$FRAMEWORK") ; then
                echo "Framework not supported yet"
                exit 1
            fi
            shift 2
            ;;
        -o|--output-file)        
            if ! [[ $2 == *.py ]] ; then
                echo "Enter a valid output path that ends with `.py`"
                exit 1
            fi
            mkdir -p "$(dirname "$2")" && cp Pipeline_handler_generalized.py $2
            #Alter the global `task` & `framework` variable
            sed -i "s/task=\".*/task=\"$TASK\"/g" $2 && sed -i "s/framework=\".*/framework=\"$FRAMEWORK\"/g" $2
            shift 2
            ;;
    esac
done