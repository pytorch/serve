#!/bin/bash
IMAGE_NAME="torchserve:1.0"

echo "Starting torchserve:1.0 docker image"

docker run -d --rm -it -p 8080:8080 -p 8081:8081 torchserve:1.0 > /dev/null 2>&1
container_id=$(docker ps --filter="ancestor=$IMAGE_NAME" -q | xargs)

sleep 30

echo "Successfully started torchserve in docker"

echo "Registering resnet-18 model"
response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar&initial_workers=1&synchronous=true")

if [ ! "$response" == 200 ]
then
    echo "failed to register model with torchserve"
else
    echo "successfully registered resnet-18 model with torchserve"
fi

echo "TorchServe is up and running with resnet-18 model"
echo "Management APIs are accessible on http://127.0.0.1:8081"
echo "Inference APIs are accessible on http://127.0.0.1:8080"
echo "For more details refer TorchServe documentation"
echo "To stop docker container for TorchServe use command : docker container stop $container_id"
