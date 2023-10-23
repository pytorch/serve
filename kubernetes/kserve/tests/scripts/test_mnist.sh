#!/usr/bin/env bash

#set -o errexit -o nounset -o pipefail

echo "MNIST test begin"

echo "Removing any previous kubernetes cluster "
minikube delete

echo "Starting kubernetes cluster "

minikube start

echo "Install kserve"

ROOT_DIR="$GITHUB_WORKSPACE"
echo "Root dir is: $ROOT_DIR "

cd $GITHUB_WORKSPACE/kserve
./hack/quick_install.sh
echo "Waiting 5s for kserve pod to come up ..."
sleep 5

echo "Deploy the cluster"

cd $GITHUB_WORKSPACE
kubectl apply -f kubernetes/kserve/tests/configs/mnist_v2_cpu.yaml

echo "Waiting 300s for pods to come up..."
sleep 300
kubectl get pods

echo "Make cluster accessible by localhost"
MODEL_NAME=mnist
SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve-mnist-v2 -o jsonpath='{.status.url}' | cut -d "/" -f 3)
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80 &

echo "Wait for ports to be in forwarding"
sleep 10

echo "Make inference request"

PREDICTION=$(curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer -d @./kubernetes/kserve/tests/data/mnist_v2.json)

EXPECTED='{"model_name":"mnist","model_version":null,"id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","parameters":null,"outputs":[{"name":"input-0","shape":[1],"datatype":"INT64","parameters":null,"data":[1]}]}'
echo "Creating a Service"

if [ "${PREDICTION}" = "${EXPECTED}" ]; then
     echo "✓ Prediction: ${PREDICTION} (Expected ${EXPECTED})"
 else
     echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
     exit 1
fi

echo "Test successful!"

echo "Delete cluster"

minikube delete
