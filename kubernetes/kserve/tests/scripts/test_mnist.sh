#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

function start_minikube_cluster() {
    echo "Removing any previous Kubernetes cluster"
    minikube delete
    echo "Starting Kubernetes cluster"
    minikube start
}

function install_kserve() {
    echo "Install Kserve"
    cd $GITHUB_WORKSPACE/kserve
    ./hack/quick_install.sh
    echo "Waiting for Kserve pod to come up ..."
    wait_for_kserve_pod 300 5
}

function deploy_cluster() {
    echo "Deploying the cluster"
    cd $GITHUB_WORKSPACE
    kubectl apply -f "$1"
    echo "Waiting for pod to come up..."
    wait_for_pod_running "$2" 300
    echo "Check status of the pod"
    kubectl get pods
    kubectl describe pod "$2"
}

function make_cluster_accessible() {
    SERVICE_NAME="$1"
    URL="$2"
    wait_for_inference_service 300 5 "$1"
    SERVICE_HOSTNAME=$(kubectl get inferenceservice ${SERVICE_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)
    wait_for_port_forwarding 5
    echo "Make inference request"
    if [ -z "$3" ]; then # for empty input data (http get method call)
        PREDICTION=$(curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" ${URL})
    else
        PREDICTION=$(curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" ${URL} -d @"$3")
    fi

    PREDICTION=$(echo -n "$PREDICTION" | tr -d '\n[:space:]')
    EXPECTED="$4"
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ SUCCESS"
        cleanup_port_forwarding
    else
        echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
        delete_minikube_cluster
        exit 1
    fi
}

function make_cluster_accessible_for_grpc() {
    PATTERN='^{.*}$' # Regex pattern to match the input value like json ex: '{"name": "mnist"}'
    PROTO_FILE_PATH="./frontend/server/src/main/resources/proto/open_inference_grpc.proto"
    SERVICE_NAME="$1"
    GRPC_METHOD="$2"
    INPUT="$3"
    wait_for_inference_service 300 5 "$1"
    SERVICE_HOSTNAME=$(kubectl get inferenceservice ${SERVICE_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)
    wait_for_port_forwarding 5
    echo "Make inference request"

    if [ -z "$INPUT" ]; then # for empty input data
        PREDICTION=$(grpcurl -plaintext -proto ${PROTO_FILE_PATH} -authority ${SERVICE_HOSTNAME} ${INGRESS_HOST}:${INGRESS_PORT} ${GRPC_METHOD})
    elif echo "$INPUT" | grep -qE "$PATTERN"; then # for pass input data with command Ex: '{"name": "mnist"}'
        PREDICTION=$(grpcurl -plaintext -d "${INPUT}" -proto ${PROTO_FILE_PATH} -authority ${SERVICE_HOSTNAME} ${INGRESS_HOST}:${INGRESS_PORT} ${GRPC_METHOD})
    else # for read input data from file
        PREDICTION=$(grpcurl -plaintext -d @ -proto ${PROTO_FILE_PATH} -authority ${SERVICE_HOSTNAME} ${INGRESS_HOST}:${INGRESS_PORT} ${GRPC_METHOD} < "${INPUT}")
    fi

    PREDICTION=$(echo -n "$PREDICTION" | tr -d '\n[:space:]')
    EXPECTED="$4"
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ SUCCESS"
        cleanup_port_forwarding
    else
        echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
        delete_minikube_cluster
        exit 1
    fi
}

function delete_minikube_cluster() {
    echo "Delete cluster"
    minikube delete
}

function wait_for_inference_service() {
    echo "Wait for inference service to be ready"
    max_wait_time="$1"
    interval="$2"
    SERVICE_NAME="$3"
    start_time=$(date +%s)
    while true; do
        service_status=$(kubectl get inferenceservice ${SERVICE_NAME} -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        if [[ "$service_status" == "True" ]]; then
            break
        fi
        current_time=$(date +%s)
        if (( current_time - start_time >= max_wait_time )); then
            echo "Timeout waiting for inference service to come up."
            delete_minikube_cluster
            exit 1
        fi
        sleep "$interval"
    done
}
function wait_for_kserve_pod() {
    max_wait_time="$1"
    interval="$2"
    start_time=$(date +%s)
    while true; do
        kserve_pod_status=$(kubectl get pods -n kserve --no-headers -o custom-columns=":status.phase")
        if [[ "$kserve_pod_status" == "Running" ]]; then
            break
        fi
        current_time=$(date +%s)
        if (( current_time - start_time >= max_wait_time )); then
            echo "Timeout waiting for Kserve pod to come up."
            delete_minikube_cluster
            exit 1
        fi
        sleep "$interval"
    done
}

function wait_for_pod_running() {
    pod_name="$1"
    max_wait_time="$2"
    interval=5
    start_time=$(date +%s)
    while true; do
        sleep "$interval"
        pod_description=$(kubectl describe pod "$pod_name")
        status_line=$(echo "$pod_description" | grep -E "Status:")
        pod_status=$(echo "$status_line" | awk '{print $2}')
        if [[ "$pod_status" == "Running" ]]; then
            break
        fi
        current_time=$(date +%s)
        if (( current_time - start_time >= max_wait_time )); then
            echo "Timeout waiting for pod $pod_name to become Running."
            delete_minikube_cluster
            exit 1
        fi
    done
}

function wait_for_port_forwarding() {
    echo "Wait for ports to be in forwarding"
    interval="$1"
    start_time=$(date +%s)
    INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
    kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80 &
    sleep "$interval"
}

function cleanup_port_forwarding() {
    echo "Clean up port forwarding"
    pkill kubectl
}

export INGRESS_HOST=localhost
export INGRESS_PORT=8080
export MODEL_NAME=mnist

start_minikube_cluster
install_kserve

echo "MNIST KServe V2 test begin"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_v2_cpu.yaml" "torchserve-mnist-v2-predictor"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer"
make_cluster_accessible "torchserve-mnist-v2" ${URL} "./kubernetes/kserve/kf_request_json/v2/mnist/mnist_v2_tensor.json" '{"model_name":"mnist","model_version":"1.0","id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","parameters":null,"outputs":[{"name":"input-0","shape":[1],"datatype":"INT64","parameters":null,"data":[1]}]}'
kubectl delete inferenceservice torchserve-mnist-v2

echo "MNIST KServe V1 test begin"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_v1_cpu.yaml" "torchserve-predictor"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict"
make_cluster_accessible "torchserve" ${URL} "./kubernetes/kserve/kf_request_json/v1/mnist.json" '{"predictions":[2]}'
kubectl delete inferenceservice torchserve

# OIP HTTP method calls
echo "MNIST Torchserve Open Inference Protocol HTTP"
SERVICE_NAME="torchserve-mnist-v2-http"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_oip_http.yaml" "torchserve-mnist-v2-http-predictor"

# ModelInfer
echo "HTTP ModelInfer method call"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer"
EXPECTED_OUTPUT='{"id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","model_name":"mnist","model_version":"1.0","outputs":[{"name":"input-0","datatype":"INT64","data":[1],"shape":[1]}]}'
make_cluster_accessible ${SERVICE_NAME} ${URL} "./kubernetes/kserve/kf_request_json/v2/mnist/mnist_v2_tensor.json" ${EXPECTED_OUTPUT}

# ServerReady
echo "HTTP ServerReady method call"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v2/health/ready"
EXPECTED_OUTPUT='{"ready":true}'
make_cluster_accessible ${SERVICE_NAME} ${URL} "" ${EXPECTED_OUTPUT}

# ServerLive
echo "HTTP ServerLive method call"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v2/health/live"
EXPECTED_OUTPUT='{"live":true}'
make_cluster_accessible ${SERVICE_NAME} ${URL} "" ${EXPECTED_OUTPUT}

# delete oip http isvc
kubectl delete inferenceservice ${SERVICE_NAME}

# OIP GRPC method calls
echo "MNIST Torchserve Open Inference Protocol GRPC"
SERVICE_NAME="torchserve-mnist-v2-grpc"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_oip_grpc.yaml" "torchserve-mnist-v2-grpc-predictor"

# ModelInfer
echo "GRPC ModelInfer method call"
GRPC_METHOD="org.pytorch.serve.grpc.openinference.GRPCInferenceService.ModelInfer"
EXPECTED_OUTPUT='{"modelName":"mnist","modelVersion":"1.0","id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","outputs":[{"name":"input-0","datatype":"INT64","shape":["1"],"contents":{"int64Contents":["1"]}}]}'
INPUT="./kubernetes/kserve/kf_request_json/v2/mnist/mnist_v2_tensor_grpc.json"
make_cluster_accessible_for_grpc "${SERVICE_NAME}" "${GRPC_METHOD}" "${INPUT}" "${EXPECTED_OUTPUT}"

# ServerReady
echo "GRPC ServerReady method call"
GRPC_METHOD="org.pytorch.serve.grpc.openinference.GRPCInferenceService.ServerReady"
EXPECTED_OUTPUT='{"ready":true}'
make_cluster_accessible_for_grpc "${SERVICE_NAME}" "${GRPC_METHOD}" "" "${EXPECTED_OUTPUT}"

# ServerLive
echo "GRPC ServerLive method call"
GRPC_METHOD="org.pytorch.serve.grpc.openinference.GRPCInferenceService.ServerLive"
EXPECTED_OUTPUT='{"live":true}'
make_cluster_accessible_for_grpc "${SERVICE_NAME}" "${GRPC_METHOD}" "" "${EXPECTED_OUTPUT}"

# ModelReady
echo "GRPC ModelReady method call"
GRPC_METHOD="org.pytorch.serve.grpc.openinference.GRPCInferenceService.ModelReady"
EXPECTED_OUTPUT='{"ready":true}'
INPUT='{"name": "mnist"}'
make_cluster_accessible_for_grpc "$SERVICE_NAME" "$GRPC_METHOD" "$INPUT" "$EXPECTED_OUTPUT"

# delete oip grpc isvc
kubectl delete inferenceservice ${SERVICE_NAME}

delete_minikube_cluster
