#!/usr/bin/env bash

# inputs:
# $1: branch name
# $2: model yaml files predefined in benchmarks/models_config/
#     - all: all yaml files in config;
#     - a list of files separated by comma: bert_multi_gpu.yaml,fastrcnn.yaml
# $3: (optional) "nightly" trigger branch installation and reports saved in AWS S3
#     Note: aws cloudwatch or s3 destination can be adjusted based on user's AWS setting.
#
# cmd examples:
# - cd serve
# - ./benchmarks/utils/benchmark_ts.sh master all nightly or ./benchmarks/utils/benchmark_ts.sh master all
# - ./benchmarks/utils/benchmark_ts.sh master bert_multi_gpu.yaml,fastrcnn.yaml nightly or
#   ./benchmarks/utils/benchmark_ts.sh master bert_multi_gpu.yaml,fastrcnn.yaml

sudo apt install -y apache2-utils

set -ex

BRANCH=$1
if [ "$BRANCH" == "" ]; then
    BRANCH=master
fi

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "CUR_DIR=$CUR_DIR"

if nvidia-smi -L; then
    hw_type=GPU
    RUNTIME="--runtime=nvidia"
    if [ "$2" == "all" ]; then
        declare -a models=("bert_multi_gpu.yaml" "fastrcnn.yaml" "mnist.yaml" "vgg16.yaml" )
    else
        IFS="," read -a models <<< $2
    fi

    echo "switch to CUDA 10.2"
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
    export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}$
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
else
    hw_type=CPU
    if [ "$2" == "all" ]; then
        declare -a models=("bert_cpu.yaml" "fastrcnn.yaml" "mnist.yaml" "vgg16.yaml" )
    else
        IFS="," read -a models <<< $2
    fi
fi

# directory to store execution log
rm -rf /tmp/ts_benchmark/
rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark

# clone TorchServe 
#if [ ! -d "serve" ]; then
#    git clone --quie https://github.com/pytorch/serve.git > /dev/null
#fi

#cd serve || exit 1

# install TorchServe
if [[ "$3" == "nightly" ]]; then
    git reset --hard
    git clean -dffx .
    git pull --rebase
    git checkout $BENCH

    if [[ "$hw_type" == "GPU" ]]; then
        python ts_scripts/install_dependencies.py --environment dev --cuda cu102
    else
        python ts_scripts/install_dependencies.py --environment dev
    fi

    python ts_scripts/install_from_src.py

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

for config_file in "$config_dir"/*; do
    echo "config_file=$config_file"
    if [ -f "$config_file" ]; then
        rm -rf ./logs
        python ./benchmarks/benchmark-ab.py --config_properties ./benchmarks/config.properties --config $config_file

	      model_name=`echo $config_file |cut -d'/' -f 3|cut -d'.' -f 1`

	      python ./benchmarks/utils/gen_metrics_json.py --csv /tmp/benchmark/ab_report.csv \
	      --log /tmp/benchmark/logs/model_metrics.log --json /tmp/benchmark/logs/model_metrics.json

	      if [ "$3" == "nightly" ]; then
            aws cloudwatch put-metric-data \
            --namespace torchserve_benchmark_${hw_type} \
            --region "us-west-2" \
            --metric-data \
            file:///tmp/benchmark/logs/model_metrics.json
        fi

	      mkdir -p /tmp/ts_benchmark/${model_name}
	      if [ -f /tmp/benchmark/ab_report.csv ]; then
            mv /tmp/benchmark/ab_report.csv /tmp/ts_benchmark/${model_name}/ab_report.csv
        fi
	      if [ -f /tmp/benchmark/predict_latency.png ]; then
            mv /tmp/benchmark/predict_latency.png /tmp/ts_benchmark/${model_name}/predict_latency.png
        fi
	      if [ -f /tmp/benchmark/result.txt ]; then
            mv /tmp/benchmark/result.txt /tmp/ts_benchmark/${model_name}/result.txt
        fi
	      if [ -f /tmp/benchmark/logs/model_metrics.log ]; then
            mv /tmp/benchmark/logs/model_metrics.log /tmp/ts_benchmark/${model_name}/model_metrics.log
        fi
        if [ -f /tmp/benchmark/logs/model_metrics.json ]; then
            mv /tmp/benchmark/logs/model_metrics.json /tmp/ts_benchmark/${model_name}/model_metrics.json
        fi
        if [ -f ./logs/model_log.log ]; then
            mv ./logs/model_log.log /tmp/ts_benchmark/${model_name}/model_log.log
        fi
        if [ -f ./logs/ts_log.log ]; then
            mv ./logs/ts_log.log /tmp/ts_benchmark/${model_name}/ts_log.log
        fi

    fi
done

# generate final report
python ./benchmarks/utils/gen_md_report.py \
  --input /tmp/ts_benchmark/ \
  --output /tmp/ts_benchmark/report.md \
  --hw ${hw_type} \
  --branch ${BRANCH}

# clean up
rm -rf json
git checkout master

# save to S3
if [[ "$3" == "nightly" ]]; then
    dt=$(date +'%Y-%m-%d')
    aws s3 cp --recursive /tmp/ts_benchmark/ s3://torchserve-model-serving/benchmark/${dt}/${hw_type,,}/
fi
date && echo "benchmark_serving.sh finished successfully."
