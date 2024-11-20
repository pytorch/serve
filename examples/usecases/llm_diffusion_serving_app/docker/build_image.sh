#!/bin/bash
set -e
BASE_IMAGE="pytorch/torchserve:latest-cpu"

DOCKER_TAG="pytorch/torchserve:llm_diffusion_serving_app"

# LLM_HF_ID=meta-llama/Meta-Llama-3-8B
# LLM_HF_ID=meta-llama/Llama-3.2-1B-Instruct
LLM_HF_ID=meta-llama/Llama-3.2-3B-Instruct
SD_HF_ID=stabilityai/stable-diffusion-xl-base-1.0

# Get relative path of example dir
EXAMPLE_DIR=$(dirname "$(readlink -f "$0")")
ROOT_DIR=${EXAMPLE_DIR}/../../../../
ROOT_DIR=$(realpath "$ROOT_DIR")
EXAMPLE_DIR=$(echo "$EXAMPLE_DIR" | sed "s|$ROOT_DIR|./|")

echo "EXAMPLE_DIR: $EXAMPLE_DIR"
echo "ROOT_DIR: $ROOT_DIR"

# Build docker image for the application
docker_build_cmd="DOCKER_BUILDKIT=1 \
docker buildx build \
--platform=linux/amd64 \
--file ${EXAMPLE_DIR}/Dockerfile \
--build-arg BASE_IMAGE=\"${BASE_IMAGE}\" \
--build-arg EXAMPLE_DIR=\"${EXAMPLE_DIR}\" \
--build-arg HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
--build-arg HTTP_PROXY=$http_proxy \
--build-arg HTTPS_PROXY=$https_proxy \
--build-arg NO_PROXY=$no_proxy \
-t \"${DOCKER_TAG}\" ."

echo -e "$docker_build_cmd"

eval $docker_build_cmd
exit_status=$?

if [ $exit_status -eq 0 ]; then
    mkdir -p model-store-local

    echo -e "\nDocker Build Successful ! \n"
    echo "............................ Next Steps ............................"
    echo "--------------------------------------------------------------------"
    echo "[Optional] Run the following command to benchmark Stable Diffusion:"
    echo "--------------------------------------------------------------------"
    echo ""
    echo "docker run --rm --platform linux/amd64 \\
        --name llm_sd_app_bench \\
        -v ${PWD}/model-store-local:/home/model-server/model-store \\
        --entrypoint python \\
        $DOCKER_TAG \\
        /home/model-server/llm_diffusion_serving_app/sd-benchmark.py -ni 3"

    echo ""
    echo "-------------------------------------------------------------------"
    echo "Run the following command to start the Multi-Image generation App:"
    echo "-------------------------------------------------------------------"
    echo ""
    echo "docker run --rm -it --platform linux/amd64 \\
        --name llm_sd_app \\
        -p 127.0.0.1:8080:8080 \\
        -p 127.0.0.1:8081:8081 \\
        -p 127.0.0.1:8082:8082 \\
        -p 127.0.0.1:8084:8084 \\
        -p 127.0.0.1:8085:8085 \\
        -v ${PWD}/model-store-local:/home/model-server/model-store \\
        -e MODEL_NAME_LLM=${LLM_HF_ID} \\
        -e MODEL_NAME_SD=${SD_HF_ID} \\
        $DOCKER_TAG"
    
    echo -e "\nNote: You can replace the model identifiers (MODEL_NAME_LLM, MODEL_NAME_SD) as needed.\n"
else
  # If docker build fails, alert the user.
  echo "Docker build failed. Try Again !"
  exit 1
fi