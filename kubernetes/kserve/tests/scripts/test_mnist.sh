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
    PREDICTION=$(curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" ${URL} -d @"$3")
    PREDICTION=$(echo -n "$PREDICTION" | tr -d '\n[:space:]')
    EXPECTED="$4"
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ SUCCESS"
        kubectl delete inferenceservice ${SERVICE_NAME}
    else
        echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
        delete_minikube_cluster
        exit 1
    fi
}

function make_cluster_accessible_for_grpc() {
    PROTO_FILE_PATH="https://raw.githubusercontent.com/andyi2it/torch-serve/oip-impl/frontend/server/src/main/resources/proto/open_inference_grpc.proto"
    curl -s -L ${PROTO_FILE_PATH} > open_inference_grpc.proto
    PROTO_FILE="open_inference_grpc.proto"
    SERVICE_NAME="$1"
    GRPC_METHOD="$2"
    wait_for_inference_service 300 5 "$1"
    SERVICE_HOSTNAME=$(kubectl get inferenceservice ${SERVICE_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)
    wait_for_port_forwarding 5
    echo "Make inference request"

    PREDICTION=$(grpcurl -plaintext -d @ -proto ${PROTO_FILE} -authority ${SERVICE_HOSTNAME} ${INGRESS_HOST}:${INGRESS_PORT} ${GRPC_METHOD} < "$3")
    PREDICTION=$(echo -n "$PREDICTION" | tr -d '\n[:space:]')
    EXPECTED="$4"
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ SUCCESS"
        kubectl delete inferenceservice ${SERVICE_NAME}
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

export INGRESS_HOST=localhost
export INGRESS_PORT=8080
export MODEL_NAME=mnist

start_minikube_cluster
install_kserve

echo "MNIST KServe V2 test begin"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_v2_cpu.yaml" "torchserve-mnist-v2-predictor"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer"
make_cluster_accessible "torchserve-mnist-v2" ${URL} "./kubernetes/kserve/kf_request_json/v2/mnist/mnist_v2_tensor.json" '{"model_name":"mnist","model_version":null,"id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","parameters":null,"outputs":[{"name":"input-0","shape":[1],"datatype":"INT64","parameters":null,"data":[1]}]}'

echo "MNIST KServe V1 test begin"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_v1_cpu.yaml" "torchserve-predictor"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict"
make_cluster_accessible "torchserve" ${URL} "./kubernetes/kserve/kf_request_json/v1/mnist.json" '{"predictions":[2]}'

echo "MNIST Torchserve Open Inference Protocol HTTP"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_oip_http.yaml" "torchserve-mnist-v2-http-predictor"
URL="http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer"
EXPECTED_OUTPUT='{"id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","model_name":"mnist","model_version":"1.0","outputs":[{"name":"input-0","datatype":"INT64","data":[1],"shape":[1]}]}'
make_cluster_accessible "torchserve-mnist-v2-http" ${URL} "./kubernetes/kserve/kf_request_json/v2/mnist/mnist_v2_tensor.json" ${EXPECTED_OUTPUT}

echo "MNIST Torchserve Open Inference Protocol GRPC"
deploy_cluster "kubernetes/kserve/tests/configs/mnist_oip_grpc.yaml" "torchserve-mnist-v2-grpc-predictor"
GRPC_METHOD="org.pytorch.serve.grpc.openinference.GRPCInferenceService.ModelInfer"
EXPECTED_OUTPUT='{"modelName":"mnist","modelVersion":"1.0","id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","outputs":[{"name":"input-0","datatype":"INT64","shape":["1"],"contents":{"int64Contents":["1"]}}]}'
make_cluster_accessible_for_grpc "torchserve-mnist-v2-grpc" ${GRPC_METHOD} "./kubernetes/kserve/kf_request_json/v2/mnist/mnist_v2_tensor_grpc.json" ${EXPECTED_OUTPUT}

delete_minikube_cluster
