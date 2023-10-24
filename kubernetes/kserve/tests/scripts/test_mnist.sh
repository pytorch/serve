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
    echo "Waiting 5s for Kserve pod to come up ..."
    wait_for_kserve_pod 300 5
}

function deploy_cluster() {
    echo "Deploying the cluster"
    cd $GITHUB_WORKSPACE
    kubectl apply -f "$1"
    echo "Waiting 120s for pod to come up..."
    wait_for_pod_running "$2" 120
    echo "Check status of the pod"
    kubectl get pods
    kubectl describe pod "$2"
}

function make_cluster_accessible() {
    MODEL_NAME="$1"
    SERVICE_NAME="$2"
    echo "Make cluster accessible by localhost"
    SERVICE_HOSTNAME=$(kubectl get inferenceservice "$SERVICE_NAME" -o jsonpath='{.status.url}' | cut -d "/" -f 3)
    export INGRESS_HOST=localhost
    export INGRESS_PORT=8080
    INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
    kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80 &
    echo "Wait for ports to be in forwarding"
    wait_for_port_forwarding 300 10
    echo "Make inference request"
    PREDICTION=$(curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer -d @"$3")
    echo "Creating a Service"
    EXPECTED="$4"
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ Prediction: ${PREDICTION} (Expected ${EXPECTED})"
    else
        echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
        exit 1
    fi
}

function delete_minikube_cluster() {
    echo "Delete cluster"
    minikube delete
}

function wait_for_kserve_pod() {
    max_wait_time="$1"
    interval="$2"
    start_time=$(date +%s)
    while true; do
        kserve_pod_status=$(kubectl get pods | grep -o "kserve" || true)
        if [[ -n "$kserve_pod_status" ]]; then
            break
        fi
        current_time=$(date +%s)
        if (( current_time - start_time >= max_wait_time )); then
            echo "Timeout waiting for Kserve pod to come up."
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
        pod_status=$(kubectl get pod "$pod_name" -o jsonpath='{.status.phase}' || true)
        if [[ "$pod_status" == "Running" ]]; then
            break
        fi
        current_time=$(date +%s)
        if (( current_time - start_time >= max_wait_time )); then
            echo "Timeout waiting for pod $pod_name to become Running."
            exit 1
        fi
        sleep "$interval"
    done
}

function wait_for_port_forwarding() {
    max_wait_time="$1"
    interval="$2"
    start_time=$(date +%s)
    while true; do
        if kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80 >/dev/null 2>&1; then
            break
        fi
        current_time=$(date +%s)
        if (( current_time - start_time >= max_wait_time )); then
            echo "Timeout waiting for port forwarding to be established."
            exit 1
        fi
        sleep "$interval"
    done
}

echo "MNIST KServe V2 test begin"
start_minikube_cluster
install_kserve
deploy_cluster "kubernetes/kserve/tests/configs/mnist_v2_cpu.yaml" "torchserve-mnist-v2-predictor"
make_cluster_accessible "mnist" "torchserve-mnist-v2" "./kubernetes/kserve/tests/data/mnist_v2.json" '{"model_name":"mnist","model_version":null,"id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","parameters":null,"outputs":[{"name":"input-0","shape":[1],"datatype":"INT64","parameters":null,"data":[1]}]}'
delete_minikube_cluster

echo "MNIST KServe V1 test begin"
start_minikube_cluster
install_kserve
deploy_cluster "kubernetes/kserve/tests/configs/mnist_v1_cpu.yaml" "torchserve-predictor"
make_cluster_accessible "mnist" "torchserve" "./kubernetes/kserve/tests/data/mnist_v1.json" '{"predictions":[2]}'
delete_minikube_cluster
