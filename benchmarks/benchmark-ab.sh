#!/usr/bin/env bash

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
        -g|--gpu)
        GPU=gpu
        shift
        ;;
        -d|--image)
        IMAGE="$2"
        shift
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
        -s|--s3)
        UPLOAD="$2"
        shift
        ;;
        --default)
        DEFAULT=YES
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if  [[ -z "${URL}" ]]; then
    echo "URL is required, for example:"
    echo "benchmark.sh -u https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar"
    echo "benchmark.sh -i lstm.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar"
    echo "benchmark.sh -c 500 -n 50000 -i noop.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    echo "benchmark.sh -d mms-cpu-local -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    echo "benchmark.sh --bsize 2 --bdelay 200 -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    exit 1
fi

echo "Preparing for benchmark..."

if [[ ! -z "${GPU}" ]] && [[ -x "$(command -v nvidia-docker)" ]]; then
    DOCKER_RUNTIME="--runtime=nvidia"
    if [[ -z "${IMAGE}" ]]; then
        IMAGE=pytorch/torchserve:latest-gpu
    fi
   ENABLE_GPU="--gpus 4"
   HW_TYPE=gpu
else
    if [[ -z "${IMAGE}" ]]; then
        IMAGE=pytorch/torchserve:latest
    fi
   ENABLE_GPU=""
   HW_TYPE=cpu
fi

docker pull "${IMAGE}"

if [[ -z "${CONCURRENCY}" ]]; then
    CONCURRENCY=1
fi

if [[ -z "${REQUESTS}" ]]; then
    REQUESTS=1000
fi

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILENAME="${URL##*/}"
MODEL="${FILENAME%.*}"

echo "Preparing config..."
rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark/conf
mkdir -p /tmp/benchmark/logs
cp -f ${BASEDIR}/config.properties /tmp/benchmark/conf/config.properties
echo "" >> /tmp/benchmark/conf/config.properties
echo "load_models=benchmark=${URL}" >> /tmp/benchmark/conf/config.properties

if [[ ! -z "${WORKER}" ]]; then
    echo "default_workers_per_model=${WORKER}" >> /tmp/benchmark/conf/config.properties
fi

echo 'setting content type'
if [[ ! -z "${INPUT}" ]] && [[ -f "${BASEDIR}/${INPUT}" ]]; then
    CONTENT_TYPE="application/json"
    cp -rf ${BASEDIR}/${INPUT} /tmp/benchmark/input
else
    CONTENT_TYPE="application/jpg"
    curl https://s3.amazonaws.com/model-server/inputs/kitten.jpg -s -S -o /tmp/benchmark/input
fi

echo "starting docker..."

 # start ts docker
set +e
docker rm -f ts
set -e
docker run ${DOCKER_RUNTIME} --name ts -p 8080:8080 -p 8081:8081 \
    -v /tmp/benchmark/conf:/opt/ml/conf \
    -v /tmp/benchmark/logs:/home/model-server/logs \
    -itd ${IMAGE} torchserve --start \
    --ts-config /opt/ml/conf/config.properties

TS_VERSION=`docker exec -it ts pip freeze | grep torchserve`
echo "ts_version is ${TS_VERSION}"

until curl -s "http://localhost:8080/ping" > /dev/null
do
  echo "Waiting for docker start..."
  sleep 3
done

container_id=$(docker ps --filter="ancestor=$IMAGE" -q | xargs)
echo "Docker started successfully with container id ${container_id}"
sleep 10

result_file="/tmp/benchmark/result.txt"
metric_log="/tmp/benchmark/logs/model_metrics.log"

if [[ -z "${BATCH_SIZE}" ]]; then
    BATCH_SIZE=1
fi

if [[ -z "${BATCH_DELAY}" ]]; then
    BATCH_DELAY=100
fi

echo "Starting Apache Bench now"

echo 'Executing inference performance test'

ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
    http://127.0.0.1:8080/predictions/benchmark > ${result_file}

echo "Apache Bench Execution completed"

echo "Grabbing performance numbers"

BATCHED_REQUESTS=$((${REQUESTS} / ${BATCH_SIZE}))
echo "requests is $REQUESTS"
echo "batch_size is $BATCH_SIZE"
echo "batched_requests is $BATCHED_REQUESTS"
line50=$((${BATCHED_REQUESTS} / 2))
line90=$((${BATCHED_REQUESTS} * 9 / 10))
line99=$((${BATCHED_REQUESTS} * 99 / 100))

grep "PredictionTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/predict.txt

MODEL_P50=`sed -n "${line50}p" /tmp/benchmark/predict.txt`
MODEL_P90=`sed -n "${line90}p" /tmp/benchmark/predict.txt`
MODEL_P99=`sed -n "${line99}p" /tmp/benchmark/predict.txt`

TS_ERROR=`grep "Failed requests:" ${result_file} | awk '{ print $NF }'`
TS_TPS=`grep "Requests per second:" ${result_file} | awk '{ print $4 }'`
TS_P50=`grep " 50\% " ${result_file} | awk '{ print $NF }'`
TS_P90=`grep " 90\% " ${result_file} | awk '{ print $NF }'`
TS_P99=`grep " 99\% " ${result_file} | awk '{ print $NF }'`
TS_MEAN=`grep -E "Time per request:.*mean\)" ${result_file} | awk '{ print $4 }'`
TS_ERROR_RATE=`echo "scale=2;100 * ${TS_ERROR}/${REQUESTS}" | bc | awk '{printf "%f", $0}'`

echo "" > /tmp/benchmark/report.txt
echo "======================================" >> /tmp/benchmark/report.txt

curl -s http://localhost:8081/models/benchmark >> /tmp/benchmark/report.txt
echo "Inference result:" >> /tmp/benchmark/report.txt
curl -s -X POST http://127.0.0.1:8080/predictions/benchmark -H "Content-Type: ${CONTENT_TYPE}" \
    -T /tmp/benchmark/input >> /tmp/benchmark/report.txt
curl -X DELETE "http://localhost:8081/models/${MODEL}"

echo "" >> /tmp/benchmark/report.txt
echo "" >> /tmp/benchmark/report.txt

echo "======================================" >> /tmp/benchmark/report.txt
echo "TS version: ${TS_VERSION}" >> /tmp/benchmark/report.txt
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

cat /tmp/benchmark/report.txt
