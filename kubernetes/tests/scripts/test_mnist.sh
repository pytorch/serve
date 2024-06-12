#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

ACCEPTABLE_CPU_CORE_USAGE=2
DOCKER_IMAGE=pytorch/torchserve-nightly:latest-gpu

# Get relative path of example dir with respect to root
# Ex: if ROOT_DIR is ~/serve , EXAMPLE_DIR is ./kubernetes/tests/scripts
EXAMPLE_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=${EXAMPLE_DIR}/../../../
ROOT_DIR=$(realpath "$ROOT_DIR")
EXAMPLE_DIR=$(echo "$EXAMPLE_DIR" | sed "s|$ROOT_DIR|./|")

function start_minikube_cluster() {
    echo "Removing any previous Kubernetes cluster"
    minikube delete
    echo "Starting Kubernetes cluster"
    minikube start --gpus  all --mount-string="$GITHUB_WORKSPACE:/host" --mount
    minikube addons enable metrics-server
}

function build_docker_image() {
    eval $(minikube docker-env)
    echo "model_api_enabled=true" >> $ROOT_DIR/$EXAMPLE_DIR/../docker/config.properties
    echo "disable_token_authorization=true" >> $ROOT_DIR/$EXAMPLE_DIR/../docker/config.properties
    docker system prune -f
    docker build -t $DOCKER_IMAGE --file $ROOT_DIR/$EXAMPLE_DIR/../docker/Dockerfile --build-arg EXAMPLE_DIR="${EXAMPLE_DIR}" .
    eval $(minikube docker-env -u)

}

function get_model_archive() {
    echo "Downloading archive for $2"
    mkdir model_store -p
    wget $1 -O model_store/"$2".mar
    pwd
    echo $GITHUB_WORKSPACE
}

function deploy_cluster() {
    echo "Deploying the cluster"
    kubectl apply -f "$1"
    echo "Waiting for pod to come up..."
    wait_for_pod_running "$2" 300
    echo "Check status of the pod"
    kubectl describe pod "$2"
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

function delete_minikube_cluster() {
    echo "Delete cluster"
    minikube delete
}

function check_cpu_cores {

  start_time=$(date +%s)
  interval=10
  while true; do
    # Check if the Metrics API error message is present
    if ! kubectl top pod -l app=$1 | grep -q $1 ; then
      sleep "$interval"
    else
      echo "Wait for metrics output to stabilize"
      sleep 60
      break
    fi
    current_time=$(date +%s)
    if (( current_time - start_time >= $2 )); then
            echo "Timeout waiting for metrics information to be available"
            delete_minikube_cluster
            exit 1
        fi
  done
  # Get the CPU cores used by the pod
  pod_name=$(kubectl get pods -l app=$1 -o json | jq -r '.items[].metadata.name')
  cpu=$(kubectl top pod -l app=$1 | awk "/${pod_name}/{print \$2}")

  # Check if the CPU cores exceed 2
  if [ $(echo "$cpu" | sed 's/m$//') -gt $ACCEPTABLE_CPU_CORE_USAGE ]; then
    echo "✘ Test failed: CPU cores $(echo "$cpu" | sed 's/m$//') for $pod_name exceeded $ACCEPTABLE_CPU_CORE_USAGE" >&2
    exit 1
  else
    echo "✓ SUCCESS"
  fi
}

function make_cluster_accessible() {
kubectl apply -f $1
kubectl port-forward svc/ts-def 8080:8080 8081:8081 &
sleep "$2"
}

function cleanup_port_forwarding() {
    echo "Clean up port forwarding"
    pkill kubectl
}

function make_prediction() {
curl -X POST "localhost:8081/models?model_name=$1&url=$1.mar&initial_workers=1"
PREDICTION=$(curl http://127.0.0.1:8080/predictions/$1 -T $2)
EXPECTED="$3"
if [ "${PREDICTION}" = "${EXPECTED}" ]; then
    echo "✓ SUCCESS"
    cleanup_port_forwarding
else
    echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
    delete_minikube_cluster
    exit 1
fi

}

# Setup
start_minikube_cluster
build_docker_image
get_model_archive "https://torchserve.pytorch.org/mar_files/mnist_v2.mar" "mnist"
deploy_cluster "./kubernetes/tests/configs/deployment.yaml" "ts-def"

echo "No model loaded CPU usage test"
check_cpu_cores "ts-def" 180

echo "MNIST test inference"
make_cluster_accessible "kubernetes/examples/mnist/service.yaml" 5
make_prediction "mnist"  "examples/image_classifier/mnist/test_data/0.png" "0"

# Clean up
delete_minikube_cluster
