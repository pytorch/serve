#!/usr/bin/env bash

IMAGE="pytorch/torchserve:latest"
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
        -b|--cnt)
	BCOUNT="$2"
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
    echo "benchmark.sh -u https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar"
    echo "benchmark.sh -i lstm.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar"
    echo "benchmark.sh -c 500 -n 50000 -i noop.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    echo "benchmark.sh -d local-image -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    exit 1
fi
echo "Preparing for benchmark..."

if [[ -z "${GPU}" ]]; then
    if [[ -z "${IMAGE}" ]]; then
        IMAGE=pytorch/torchserve:latest
    fi
   ENABLE_GPU=""
   HW_TYPE=cpu
else
    DOCKER_RUNTIME="--runtime=nvidia"
    if [[ -z "${IMAGE}" ]]; then
        IMAGE=pytorch/torchserve:latest-gpu
    fi
   ENABLE_GPU="--gpus 4"
   HW_TYPE=gpu
fi

docker pull "${IMAGE}"

if [[ -z "${CONCURRENCY}" ]]; then
    CONCURRENCY=100
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
if [[ ! -z "${WORKER}" ]]; then
    echo "default_workers_per_model=${WORKER}" >> /tmp/benchmark/conf/config.properties
fi

if [[ -z "${OP}" ]] || test "${OP}" = "R"; then
    #echo "load_models=benchmark=${URL}" >> /tmp/benchmark/conf/config.properties
    echo 'setting content type'
    if [[ ! -z "${INPUT}" ]] && [[ -f "${BASEDIR}/${INPUT}" ]]; then
        CONTENT_TYPE="application/jpg"
        cp -rf ${BASEDIR}/${INPUT} /tmp/benchmark/input
    else
        CONTENT_TYPE="application/jpg"
        curl https://s3.amazonaws.com/model-server/inputs/kitten.jpg -s -S -o /tmp/benchmark/input
    fi
fi

echo "starting docker..."

 #start ts docker
set +e
echo "docker_runtime is ${DOCKER_RUNTIME} and enable_gpu is ${ENABLE_GPU} and image is ${IMAGE}"
if [ ! "$(docker ps -q -f name="ts")" ]; then
    if [ "$(docker ps -aq -f status=exited -f name="ts")" ]; then
        # cleanup
        docker rm -f ts
    fi
    # run your container
    docker run ${DOCKER_RUNTIME} ${ENABLE_GPU} --name ts -p 8080:8080 -p 8081:8081 \
        -v /tmp/benchmark/conf:/home/ubuntu/model-server \
        -v /tmp/benchmark/logs:/home/ubuntu/model-server/logs \
        -u root -itd ${IMAGE} torchserve --start \
        --ts-config /home/model-server/config.properties
fi
set -e
#docker run -d --rm -it --name ts -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest > /dev/null 2>&1
#    -v /tmp/benchmark/conf:/opt/ml/conf \
#    -v /tmp/benchmark/logs:/home/model-server/logs \
#    -u root torchserve --start --ncs\
#    --ts-config /opt/ml/conf/config.properties\
#> /dev/null 2>&1
#docker run ${DOCKER_RUNTIME} --name ts --rm -p 8080:8080 -p 8081:8081 $ENABLE_GPU\
#    -v /tmp/benchmark/conf:/opt/ml/conf \
#    -v /tmp/benchmark/logs:/home/model-server/logs \
#    -u root -itd ${IMAGE} torchserve --start --ncs\
#    --ts-config /opt/ml/conf/config.properties

echo "2.2.2.2"
TS_VERSION=`docker exec -it ts pip freeze | grep torchserve`
echo "ts_version is ${TS_VERSION}"
set -e
echo "3333"
until curl -s "http://localhost:8080/ping" > /dev/null
do
  echo "Waiting for docker start..."
  sleep 3
done

echo "4444"
echo "Docker started successfully"

sleep 10
container_id=$(docker ps --filter="ancestor=$IMAGE" -q | xargs)
echo "Docker initiated with container id ${container_id}"

if [[ -z ${OP} ]] || test "${OP}" = "R"; then
   echo "model already registered so should not send register request again"
   sleep 10
else
    echo "http://localhost:8081/models?url=${URL}&initial_workers=1&synchronous=true"
    response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=${URL}&initial_workers=1&synchronous=true")

    if [ ! "$response" == 200 ]
    then
        echo "failed to register model with torchserve"
    else
        echo "successfully registered the model with torchserve"
    fi
fi

result_file="/tmp/benchmark/result.txt"
metric_log="/tmp/benchmark/logs/model_metrics.log"

echo "Executing ab"

if [[ -z "${OP}" ]]; then
    echo 'Executing inference performance test'
    ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
        http://127.0.0.1:8080/predictions/${MODEL} > ${result_file}

    if [[ -z "${BATCH_SIZE}" ]]; then
	    BATCH_SIZE=1
    fi
else
    echo "Executing operation ${OP}"
    
    if [[ -z "${BATCH_SIZE}" ]]; then
	    BATCH_SIZE=1
    fi

    if [[ -z "${BATCH_DELAY}" ]]; then
	    BATCH_DELAY=100
    fi

    if [[ -z "${WORKERS}" ]]; then
	    WORKERS=1
    fi

    if test "${OP}" = "R"; then	
        RURL="?model_name=${MODEL}&url=${URL}&batch_size=${BATCH_SIZE}&max_batch_delay=${BATCH_DELAY}&initial_workers=${WORKERS}&synchronous=true"
        curl -X POST "http://localhost:8081/models${RURL}"

	echo 'Executing inference performance test'
        ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
        http://127.0.0.1:8080/predictions/${MODEL} > ${result_file}

	OP=""
    fi
fi

echo "ab Execution completed"

echo "Grabing performance numbers"

BATCHED_REQUESTS=$((${REQUESTS} / ${BATCH_SIZE}))
echo "requests is $REQUESTS"
echo "batch_size is $BATCH_SIZE"
echo "batched_requests is $BATCHED_REQUESTS"
line50=$((${BATCHED_REQUESTS} / 2))
line90=$((${BATCHED_REQUESTS} * 9 / 10))
line99=$((${BATCHED_REQUESTS} * 99 / 100))

if [[ -z "${OP}" ]] || test "${OP}" = "R"; then
    grep "PredictionTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/predict.txt
    grep "PreprocessTime" ${metric_log} | cut -c55- | cut -d"|" -f1 | sort -g > /tmp/benchmark/preprocess.txt
    grep "InferenceTime" ${metric_log} | cut -c54- | cut -d"|" -f1 | sort -g > /tmp/benchmark/inference.txt
    grep "PostprocessTime" ${metric_log} | cut -c56- | cut -d"|" -f1 | sort -g > /tmp/benchmark/postprocess.txt

    MODEL_P50=`sed -n "${line50}p" /tmp/benchmark/predict.txt`
    MODEL_P90=`sed -n "${line90}p" /tmp/benchmark/predict.txt`
    MODEL_P99=`sed -n "${line99}p" /tmp/benchmark/predict.txt`
fi

TS_ERROR=`grep "Failed requests:" ${result_file} | awk '{ print $NF }'`
TS_TPS=`grep "Requests per second:" ${result_file} | awk '{ print $4 }'`
TS_P50=`grep " 50\% " ${result_file} | awk '{ print $NF }'`
TS_P90=`grep " 90\% " ${result_file} | awk '{ print $NF }'`
TS_P99=`grep " 99\% " ${result_file} | awk '{ print $NF }'`
TS_MEAN=`grep -E "Time per request:.*mean\)" ${result_file} | awk '{ print $4 }'`
TS_ERROR_RATE=`echo "scale=2;100 * ${TS_ERROR}/${REQUESTS}" | bc | awk '{printf "%f", $0}'`

echo "" > /tmp/benchmark/report.txt
echo "======================================" >> /tmp/benchmark/report.txt

if [[ -z "${OP}" ]] || test "${OP}" = "R"; then
    curl -s http://localhost:8081/models/${MODEL} >> /tmp/benchmark/report.txt
    echo "Inference result:" >> /tmp/benchmark/report.txt
    curl -s -X POST http://127.0.0.1:8080/predictions/${MODEL} -H "Content-Type: ${CONTENT_TYPE}" \
        -T /tmp/benchmark/input >> /tmp/benchmark/report.txt
    echo "DB.. Now unregistering model"
    #curl -X DELETE "http://localhost:8081/models/${MODEL}"
else
    echo "Benchmark results - Management API - ${OP}"
fi

echo "" >> /tmp/benchmark/report.txt
echo "" >> /tmp/benchmark/report.txt

echo "======================================" >> /tmp/benchmark/report.txt
echo "TS version: ${TS_VERSION}" >> /tmp/benchmark/report.txt
echo "CPU/GPU: ${HW_TYPE}" >> /tmp/benchmark/report.txt
echo "Model: ${MODEL}" >> /tmp/benchmark/report.txt
echo "Concurrency: ${CONCURRENCY}" >> /tmp/benchmark/report.txt
echo "Requests: ${REQUESTS}" >> /tmp/benchmark/report.txt

if [[ -z "${OP}" ]] || test "${OP}" = "R"; then
    echo "Model latency P50: ${MODEL_P50}" >> /tmp/benchmark/report.txt
    echo "Model latency P90: ${MODEL_P90}" >> /tmp/benchmark/report.txt
    echo "Model latency P99: ${MODEL_P99}" >> /tmp/benchmark/report.txt
fi
echo "TS throughput: ${TS_TPS}" >> /tmp/benchmark/report.txt
echo "TS latency P50: ${TS_P50}" >> /tmp/benchmark/report.txt
echo "TS latency P90: ${TS_P90}" >> /tmp/benchmark/report.txt
echo "TS latency P99: ${TS_P99}" >> /tmp/benchmark/report.txt
echo "TS latency mean: ${TS_MEAN}" >> /tmp/benchmark/report.txt
echo "TS error rate: ${TS_ERROR_RATE}%" >> /tmp/benchmark/report.txt

cat /tmp/benchmark/report.txt
#set +x
