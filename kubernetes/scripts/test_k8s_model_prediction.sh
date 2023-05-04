#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

echo "Creating mnist.mar file "

torch-model-archiver --model-name mnist \
     --version 1.0 \
     --model-file examples/image_classifier/mnist/mnist.py \
     --serialized-file examples/image_classifier/mnist/mnist_cnn.pt \
     --handler  examples/image_classifier/mnist/mnist_handler.py

mkdir -p model_store
mv -f mnist.mar model_store/

echo "Starting kubernetes cluster "

minikube start --mount-string="$HOME/serve:/host" --mount

echo "Deploy the cluster"

kubectl apply -f kubernetes/examples/mnist/deployment.yaml

echo "Waiting 120s for pods to come up..."
sleep 120

echo "Creating a Service"

kubectl apply -f kubernetes/examples/mnist/service.yaml

echo "Waiting 10s for service to come up..."
sleep 10

echo "Make cluster accessible by localhost"

kubectl port-forward svc/ts-def 8080:8080 8081:8081 &

sleep 5

echo "Register MNIST model"

curl -X POST "localhost:8081/models?model_name=mnist&url=mnist.mar&initial_workers=4"

assert_expected() {
    PREDICTION=$1
    EXPECTED=$2
    if [ "${PREDICTION}" = "${EXPECTED}" ]; then
        echo "✓ Prediction: ${PREDICTION} (Expected ${EXPECTED})"
    else
        echo "✘ Test failed: Prediction: ${PREDICTION}, expected ${EXPECTED}."
        exit 1
   fi
}

echo "Testing classifier with test images"
for EXPECTED in {0..9}
do
    PREDICTION=$(curl -s localhost:8080/predictions/mnist -T "examples/image_classifier/mnist/test_data/${EXPECTED}.png")
    assert_expected "${PREDICTION}" "${EXPECTED}"
done

echo "Test successful!"

echo "Delete cluster"

minikube delete
