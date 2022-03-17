#!/usr/bin/env bash

# inputs:
# $1: branch name or nightly build
#     - nightly: install torchserve-nightly
#     - branch name: install the branch
#     - skip: skip installation
# $2: model yaml files predefined in benchmarks/models_config/
#     - all: all yaml files in config;
#     - a list of files separated by comma: bert_multi_gpu.yaml,fastrcnn.yaml
# $3: (optional) "S3" trigger reports and metrics0 saved in AWS S3
#
# Note:
# - aws cloudwatch or s3 destination can be adjusted based on user's AWS setting.
# - aws cloudwatch metric-data max size 40kb
#
# cmd examples:
# - cd serve
# - ./benchmarks/utils/benchmark_ts.sh master all s3 or ./benchmarks/utils/benchmark_ts.sh master all
# - ./benchmarks/utils/benchmark_ts.sh nightly bert_multi_gpu.yaml,fastrcnn.yaml s3 or
#   ./benchmarks/utils/benchmark_ts.sh skip bert_multi_gpu.yaml,fastrcnn.yaml

sudo apt install -y apache2-utils

set -ex

BRANCH=$1
if [ "$BRANCH" == "" ]; then
    BRANCH=master
fi

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "CUR_DIR=$CUR_DIR"

if [ "$2" == "all" ]; then
    declare -a models=("bert_cpu.yaml" "mnist.yaml")
    #declare -a models=`ls ./benchmarks/models_config | sed 's,\(.*\),"\1",' |  tr '\n' ' '`
    #declare -a models=`ls ./benchmarks/models_config | tr '\n' ' '`
else
    IFS="," read -a models <<< $2
fi

if nvidia-smi -L; then
    hw_type=GPU

    echo "switch to CUDA 10.2"
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
    export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}$
else
    hw_type=CPU
fi

# directory to store execution log
rm -rf /tmp/ts_benchmark/
rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark

# clone TorchServe 
#if [ ! -d "serve" ]; then
#    git clone --quiet https://github.com/pytorch/serve.git > /dev/null
#fi

#cd serve || exit 1

# install TorchServe
if [[ "$BRANCH" != "skip" ]]; then
    echo "not skip\n"
    if [[ "$hw_type" == "GPU" ]]; then
        python ts_scripts/install_dependencies.py --environment dev --cuda cu102
    else
        python ts_scripts/install_dependencies.py --environment dev
    fi

    if [[ "$BRANCH" == "nightly" ]]; then
        pip install torchserve-nightly
    else
        git reset --hard
        git clean -dffx .
        git pull --rebase
        git checkout $BRANCH
        python ts_scripts/install_from_src.py
    fi

    pip install -r benchmarks/requirements-ab.txt
fi


# generate benchmark json config files
rm -rf json
mkdir -p json/cpu
mkdir -p json/gpu

for model in "${models[@]}"; do
    input_file="./benchmarks/models_config/${model}"
    echo "input_file=$input_file"
    python ./benchmarks/utils/gen_model_config_json.py --input $input_file --output json
done

# run benchmark
config_dir="json/cpu"
if [ "$hw_type" == "GPU" ]; then
    config_dir="json/gpu"
fi

declare -a result_files=("ab_report.csv" "predict_latency.png")
declare -a log_files=("model_log.log" "ts_log.log")
for config_file in "$config_dir"/*; do
    echo "config_file=$config_file"
    if [ -f "$config_file" ]; then
        rm -rf ./logs
        python ./benchmarks/benchmark-ab.py --config_properties ./benchmarks/config.properties --config $config_file

	      model_name=`echo $config_file |cut -d'/' -f 3|cut -d'.' -f 1`

	      python ./benchmarks/utils/gen_metrics_json.py --csv /tmp/benchmark/ab_report.csv \
	      --log /tmp/benchmark/logs/model_metrics.log \
	      --stats /tmp/benchmark/logs/stats_metrics.json \
	      --raw /tmp/benchmark/logs/model_metrics.json

	      if [ "$3" == "s3" ]; then
            aws cloudwatch put-metric-data \
            --namespace torchserve_benchmark_${hw_type} \
            --region "us-west-2" \
            --metric-data \
            file:///tmp/benchmark/logs/stats_metrics.json
        fi

        mkdir -p /tmp/ts_benchmark/${model_name}
	      for resulte_file in ${result_files[@]}; do
	          if [ -f /tmp/benchmark/"$resulte_file" ]; then
	              mv /tmp/benchmark/"$resulte_file" /tmp/ts_benchmark/"${model_name}"/"$resulte_file"
	          fi
	      done

	      for log_file in ${$log_files[@]}; do
	          if [ -f ./logs/"$log_file" ]; then
	              mv ./logs/"$log_file" /tmp/ts_benchmark/"${model_name}"/"$log_file"
	          fi
	      done
    fi
done

# generate final report
python ./benchmarks/utils/gen_md_report.py \
  --input /tmp/ts_benchmark/ \
  --output /tmp/ts_benchmark/report.md \
  --hw ${hw_type} \
  --branch ${BRANCH}

# clean up
#rm -rf json
#git checkout master

# save to S3
if [[ "$3" == "s3" ]]; then
    dt=$(date +'%Y-%m-%d')
    aws s3 cp --recursive /tmp/ts_benchmark/ s3://torchserve-model-serving/benchmark/${dt}/${hw_type,,}/
fi
date && echo "benchmark_serving.sh finished successfully."
