#!/usr/bin/env bash

#set -ex
set -e

POSITIONAL=()

while [[ $# -gt 0 ]]
do
    key="$1"
    case ${key} in
        -u|--url)
        URL="$2"
        shift
        shift
        ;;
        -m|--model)
        MODEL="$2"
        shift
        shift
        ;;
        -g|--gpu)
        GPU=gpu
        shift
        ;;
        -c|--concurrency)
        CONCURRENCY="$2"
        shift
        shift
        ;;
        -n|--requests)
        REQUESTS="$2"
        shift
        shift
        ;;
        -i|--input)
        INPUT="$2"
        shift
        shift
        ;;
       -w|--worker)
        WORKER="$2"
        shift
        shift
        ;;
        --bdelay)
        BATCH_DELAY="$2"
        shift
        shift
        ;;
        --bsize)
        BATCH_SIZE="$2"
        shift
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [[ -z "${MODEL}" ]] && [[ -z "${URL}" ]]; then
    echo "URL / MODEL is required, for example:"
    echo "benchmark-ab.sh --url https://torchserve-mar-files.s3.amazonaws.com/vgg11.mar --bsize 1 --bdelay 50 --worker 4 --input kitten.jpg --requests 1000 --concurrency 100"
    exit 1
fi

if [[ -z "${GPU}" ]]; then
   ENABLE_GPU=""
   HW_TYPE=cpu
else
   ENABLE_GPU="--gpus 4"
   HW_TYPE=gpu
fi

if [[ -z "${CONCURRENCY}" ]]; then
    CONCURRENCY=1
fi

if [[ -z "${REQUESTS}" ]]; then
    REQUESTS=1000
fi

if [[ -z "${BATCH_SIZE}" ]]; then
    BATCH_SIZE=10
fi

if [[ -z "${BATCH_DELAY}" ]]; then
    BATCH_DELAY=50
fi

if [[ -z "${WORKER}" ]]; then
    WORKER=1
fi


torchserve --stop

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Preparing config..."
mkdir -p /tmp/model_store
rm -rf /tmp/model_store/*
rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark/conf
mkdir -p /tmp/benchmark/logs
cp -f ${BASEDIR}/config.properties /tmp/benchmark/conf/config.properties
cp $INPUT /tmp/benchmark/input
echo "" >> /tmp/benchmark/conf/config.properties
if [[ ! -z "${WORKER}" ]]; then
    echo "default_workers_per_model=${WORKER}" >> /tmp/benchmark/conf/config.properties
fi

echo "starting torchserve..."

torchserve --model-store /tmp/model_store --ts-config /tmp/benchmark/conf/config.properties  &> /tmp/benchmark/logs/model_metrics.log

until curl -s "http://localhost:8080/ping" > /dev/null
do
  echo "Waiting for torchserve to start..."
  sleep 3
done

echo "torchserve started successfully"

sleep 10

result_file="/tmp/benchmark/result.txt"
metric_log="/tmp/benchmark/logs/model_metrics.log"

echo "Registering model ..."

RURL="?model_name=${MODEL}&url=${URL}&batch_size=${BATCH_SIZE}&max_batch_delay=${BATCH_DELAY}&initial_workers=${WORKERS}&synchronous=true"
curl -X POST "http://localhost:8081/models${RURL}"

echo "Executing Apache Bench tests ..."

echo 'Executing inference performance test'
ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
http://127.0.0.1:8080/predictions/${MODEL} > ${result_file}

echo "Unregistering model ..."
sleep 10
curl -X DELETE "http://localhost:8081/models/${MODEL}"
sleep 10
echo "Execution completed"
torchserve --stop
echo "Torchserve stopped"

echo "Grabing performance numbers"

BATCHED_REQUESTS=$((${REQUESTS} / ${BATCH_SIZE}))
line50=$((${BATCHED_REQUESTS} / 2))
line90=$((${BATCHED_REQUESTS} * 9 / 10))
line99=$((${BATCHED_REQUESTS} * 99 / 100))

grep "PredictionTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/predict.txt
grep "PreprocessTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/preprocess.txt
grep "InferenceTime" ${metric_log} | cut -c54- | cut -d"|" -f1 | sort -g > /tmp/benchmark/inference.txt
grep "PostprocessTime" ${metric_log} | cut -c56- | cut -d"|" -f1 | sort -g > /tmp/benchmark/postprocess.txt

MODEL_P50=`sed -n "${line50}p" /tmp/benchmark/predict.txt`
MODEL_P90=`sed -n "${line90}p" /tmp/benchmark/predict.txt`
MODEL_P99=`sed -n "${line99}p" /tmp/benchmark/predict.txt`
MODEL_P50=$(echo ${MODEL_P50}| cut -d':'  -f2)
MODEL_P90=$(echo ${MODEL_P90}| cut -d':'  -f2)
MODEL_P99=$(echo ${MODEL_P99}| cut -d':'  -f2)


TS_ERROR=`grep "Failed requests:" ${result_file} | awk '{ print $NF }'`
TS_TPS=`grep "Requests per second:" ${result_file} | awk '{ print $4 }'`
TS_P50=`grep " 50\% " ${result_file} | awk '{ print $NF }'`
TS_P90=`grep " 90\% " ${result_file} | awk '{ print $NF }'`
TS_P99=`grep " 99\% " ${result_file} | awk '{ print $NF }'`
TS_MEAN=`grep -E "Time per request:.*mean\)" ${result_file} | awk '{ print $4 }'`
TS_ERROR_RATE=`echo "scale=2;100 * ${TS_ERROR}/${REQUESTS}" | bc | awk '{printf "%f", $0}'`

echo "" > /tmp/benchmark/report.txt
echo "CPU/GPU: ${HW_TYPE}" >> /tmp/benchmark/report.txt
echo "Model: ${MODEL}" >> /tmp/benchmark/report.txt
echo "Concurrency: ${CONCURRENCY}" >> /tmp/benchmark/report.txt
echo "Requests: ${REQUESTS}" >> /tmp/benchmark/report.txt
echo "Model latency P50: ${MODEL_P50}" >> /tmp/benchmark/report.txt
echo "Model latency P90: ${MODEL_P90}" >> /tmp/benchmark/report.txt
echo "Model latency P99: ${MODEL_P99}" >> /tmp/benchmark/report.txt
echo "TS throughput: ${TS_TPS}" >> /tmp/benchmark/report.txt
echo "TS latency P50: ${TS_P50}" >> /tmp/benchmark/report.txt
echo "TS latency P90: ${TS_P90}" >> /tmp/benchmark/report.txt
echo "TS latency P99: ${TS_P99}" >> /tmp/benchmark/report.txt
echo "TS latency mean: ${TS_MEAN}" >> /tmp/benchmark/report.txt
echo "TS error rate: ${TS_ERROR_RATE}%" >> /tmp/benchmark/report.txt

echo "CSV : ${HW_TYPE}, ${MODEL}, ${BATCH_SIZE}, ${CONCURRENCY}, ${REQUESTS}, ${MODEL_P50}, ${MODEL_P90}, ${MODEL_P99}, ${TS_TPS}, ${TS_P50}, ${TS_P90}, ${TS_P90}, ${TS_P99}, ${TS_MEAN}, ${TS_ERROR_RATE}" >> /tmp/benchmark/report.txt

cat /tmp/benchmark/report.txt
