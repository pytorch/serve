#!/bin/sh
TASK_NAME="CoLA"
MODEL_TYPE="bert"
CHECKPOINT_NAME="bert-base-cased"
OUTPUT_DIR="./outputs"

python download_glue_data.py --tasks=${TASK_NAME}


python run_glue.py \
    --model_type "${MODEL_TYPE}" \
    --model_name_or_path "${CHECKPOINT_NAME}" \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir "glue_data/${TASK_NAME}" \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir "${OUTPUT_DIR}"
