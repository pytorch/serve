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
        -o|--op)
        OP="$2"
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

if [[ -z "${OP}" ]] && [[ -z "${URL}" ]]; then
    echo "URL is required, for example:"
    echo "benchmark.sh -u https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar"
    echo "benchmark.sh -i lstm.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar"
    echo "benchmark.sh -c 500 -n 50000 -i noop.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    echo "benchmark.sh -d local-image -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar"
    exit 1
fi
echo "Preparing for benchmark..."

if [[ -z "${OP}" ]]; then
    echo "URL is mandatory and it should be any management api url."
    exit 1
fi

#if [[ -x "$(command -v nvidia-docker)" ]]; then
#    GPU=true
#else
#    GPU=false
#fi

if [[ -z "${GPU}" ]]; then
   ENABLE_GPU=""
   HW_TYPE=cpu
else
   ENABLE_GPU="--gpus 4"
   HW_TYPE=gpu
fi

#if [[ "${GPU}" == "true" ]]; then
#    DOCKER_RUNTIME="--runtime=nvidia"
#    if [[ -z "${IMAGE}" ]]; then
#        IMAGE=awsdeeplearningteam/multi-model-server:nightly-mxnet-gpu
#        docker pull "${IMAGE}"
#    fi
#    #HW_TYPE=gpu
#    #ENABLE_GPU="--gpus 4"
#else
#    if [[ -z "${IMAGE}" ]]; then
#        IMAGE=awsdeeplearningteam/multi-model-server:nightly-mxnet-cpu
#        docker pull "${IMAGE}"
#    fi
    #HW_TYPE=cpu
#fi

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

# start ts docker
#set +e
#docker rm -f ts
#set -e
#docker run ${DOCKER_RUNTIME} --name ts -p 8080:8080 -p 8081:8081 $ENABLE_GPU\
#    -v /tmp/benchmark/conf:/opt/ml/conf \
#    -v /tmp/benchmark/logs:/home/model-server/logs \
#    -u root -itd ${IMAGE} torchserve --start --ncs\
#    --ts-config /opt/ml/conf/config.properties

#echo "Docker initiated"
#
#TS_VERSION=`docker exec -it ts pip freeze | grep torchserve`
#
until curl -s "http://localhost:8080/ping" > /dev/null
do
  echo "Waiting for docker start..."
  sleep 3
done

echo "Docker started successfully"

sleep 10

result_file="/tmp/benchmark/result.txt"
metric_log="/tmp/benchmark/logs/model_metrics.log"

echo "Executing ab"

if [[ -z "${OP}" ]]; then
    echo 'Executing inference performance test'
    ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
        http://127.0.0.1:8080/predictions/${MODEL} > ${result_file}
else
    echo "Executing operation ${OP}"
    
    if [[ -z "${BATCH_SIZE}" ]]; then
	    BATCH_SIZE=1
    fi

    if [[ -z "${BATCH_DELAY}" ]]; then
	    BATCH_DELAY=100
    fi

    if test "${OP}" = "R"; then	
        RURL="?model_name=${MODEL}&url=${URL}&batch_size=${BATCH_SIZE}&max_batch_delay=${BATCH_DELAY}&initial_workers=${WORKERS}&synchronous=true"
        curl -X POST "http://localhost:8081/models${RURL}"

	echo 'Executing inference performance test'
        ab -c ${CONCURRENCY} -n ${REQUESTS} -k -p /tmp/benchmark/input -T "${CONTENT_TYPE}" \
        http://127.0.0.1:8080/predictions/${MODEL} > ${result_file}

	OP=""
    fi
    
    if test "${OP}" = "D"; then
        ab -s 180 -c ${CONCURRENCY} -n ${REQUESTS} -k -m DELETE "http://localhost:8081/models/${MODEL}" > ${result_file}
    fi

    if test "${OP}" = "SF"; then
            ab -s 180 -c ${CONCURRENCY} -n ${REQUESTS} -k -m PUT "http://localhost:8081/models/${MODEL}/1_0/set-default" > ${result_file}
    fi

    if test "${OP}" = "U"; then
            ab -s 180 -c ${CONCURRENCY} -n ${REQUESTS} -k -m PUT "http://localhost:8081/models/${MODEL}?min_worker=${WORKERS}&synchronous=true" > ${result_file}
    fi

    #done
fi

echo "ab Execution completed"

echo "Grabing performance numbers"

BATCHED_REQUESTS=$((${REQUESTS} / ${BATCH_SIZE}))
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
    curl -X DELETE "http://localhost:8081/models/${MODEL}"
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

if [[ ! -z "${UPLOAD}" ]]; then
    TODAY=`date +"%y-%m-%d_%H"`
    echo "Saving on S3 bucket on s3://benchmarkai-metrics-prod/daily/ts/${HW_TYPE}/${TODAY}/${MODEL}"

    aws s3 cp /tmp/benchmark/ s3://benchmarkai-metrics-prod/daily/ts/${HW_TYPE}/${TODAY}/${MODEL} --recursive

    echo "Files uploaded"
fi

#set +x
