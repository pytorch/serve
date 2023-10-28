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
    wait_for_pod_running "$2" 120
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
    EXPECTED="$4"
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ SUCCESS"
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

delete_minikube_cluster
