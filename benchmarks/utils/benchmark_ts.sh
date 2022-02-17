#!/usr/bin/env bash

sudo apt install -y apache2-utils

set -ex

BRANCH=$1
if [[ "$BRANCH" == "" ]]; then
    BRANCH=master
fi

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "CUR_DIR=$CUR_DIR"

date

if nvidia-smi -L; then
    hw_type=GPU
    RUNTIME="--runtime=nvidia"
    declare -a models=("bert_multi_gpu.yaml" "fastrcnn.yaml" "mnist.yaml" "vgg16.yaml")

    echo "switch to CUDA 10.2"
    sudo rm -rf /usr/local/cuda
    sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
    export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}$
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
else
    hw_type=CPU
    declare -a models=("vgg16.yaml")
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
if [[ "$2" == "nightly" ]]; then
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

    python ts_scripts/install_dependencies.py --environment dev 

    pip install -r benchmarks/requirements-ab.txt
    pip install -r benchmarks/automated/requirements.txt
fi

# generate benchmark json config files
config_dir="json/cpu"
if [ "$hw_type" == "GPU" ]; then
    config_dir="json/gpu"
fi

rm -rf ${config_dir}
mkdir -p ${config_dir}

for model in "${models[@]}"; do
    input_file="./benchmarks/automated/tests/suite/${model}"
    echo "input_file=$input_file"
    python ./benchmarks/utils/gen_model_config_json.py --input $input_file --output json
done

# run benchmark
for config_file in "$config_dir"/*; do
    echo "config_file=$config_file"
    if [ -f "$config_file" ]; then
        python ./benchmarks/benchmark-ab.py --config_properties ./benchmarks/config.properties --config $config_file

	      model_name=`echo $config_file |cut -d'/' -f 3|cut -d'.' -f 1`

	      serving_metrics=`python ./benchmarks/utils/gen_metrics_json.py --input /tmp/benchmark/ab_report.csv`
	      if [ "$2" == "nightly" ]; then
            aws cloudwatch put-metric-data --namespace "torchserve_benchmark_${hw_type}" --region "us-west-2" --metric-data "$serving_metrics"
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
    aws s3 cp --recursive /tmp/ts_benchmark/ "s3://torchserve-model-serving/benchmark/${dt}/${hw_type,,}/"
fi
date && echo "benchmark_serving.sh finished successfully."
