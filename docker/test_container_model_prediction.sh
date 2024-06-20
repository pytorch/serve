#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

IMAGE_TAG=$1
CONTAINER="test-container-${IMAGE_TAG}"

FILES_PATH="$(realpath "$(pwd)/..")/examples/image_classifier/mnist"
SERVER_PATH="/home/model-server"
TEST_ENTRYPOINT="$(pwd)/test-entrypoint.sh"
cat << EOF > "${TEST_ENTRYPOINT}"
#!/usr/bin/env bash

torch-model-archiver \
    --model-name=mnist \
    --version=1.0 \
    --model-file=/home/model-server/mnist.py \
    --serialized-file=/home/model-server/mnist_cnn.pt \
    --handler=/home/model-server/mnist_handler.py \
    --export-path=/home/model-server/model-store

torchserve --start --ts-config=/home/model-server/config.properties --models mnist=mnist.mar --disable-token
EOF

echo "Starting container ${CONTAINER}"
docker run --rm -d -it --name "${CONTAINER}" -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 \
    -v "${FILES_PATH}/mnist.py":"${SERVER_PATH}/mnist.py" \
    -v "${FILES_PATH}/mnist_cnn.pt":"${SERVER_PATH}/mnist_cnn.pt" \
    -v "${FILES_PATH}/mnist_handler.py":"${SERVER_PATH}/mnist_handler.py" \
    -v "${TEST_ENTRYPOINT}":"${SERVER_PATH}/test-entrypoint.sh" \
    "${IMAGE_TAG}" \
    /bin/bash test-entrypoint.sh

echo "Waiting 10s for container to come up..."
sleep 10

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

echo "Testing classifier with test images in container ${CONTAINER}..."
for EXPECTED in {0..9}
do
    PREDICTION=$(curl -s localhost:8080/predictions/mnist -T "${FILES_PATH}/test_data/${EXPECTED}.png")
    assert_expected "${PREDICTION}" "${EXPECTED}"
done

echo "Test successful!"

docker stop "${CONTAINER}"
rm -f "${TEST_ENTRYPOINT}"
