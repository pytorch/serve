
# Multi-Image Generation Streamlit App by chaining Llama & Stable Diffusion using TorchServe, torch.compile & OpenVINO

This Streamlit app is designed to generate multiple images based on a provided text prompt. It leverages [TorchServe](https://pytorch.org/serve/) for efficient model serving and management, and utilizes [Meta-LLaMA-3.2](https://huggingface.co/meta-llama) for prompt generation, and **Stable Diffusion** with [latent-consistency/lcm-sdxl](https://huggingface.co/latent-consistency/lcm-sdxl) and [Torch.compile using OpenVINO backend](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) for image generation.

![Multi-Image Generation App Workflow](./docker/workflow-1.png)

## Quick Start Guide

**Prerequisites**: 
- Docker installed on your system
- Hugging Face Token: Create a Hugging Face account and obtain a token with access to the [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model.


To launch the app, you need to run the following:
```bash
# 1: Set HF Token as Env variable
export HUGGINGFACE_TOKEN=<HUGGINGFACE_TOKEN>

# 2: Build Docker image of this Multi-Image Generation App
git clone https://github.com/pytorch/serve.git
cd serve
./examples/usecases/llm_diffusion_serving_app/docker/build_image.sh

# 3: Launch the streamlit app for server & client
# After the Docker build is successful, you will see a command printed to start the app. Run that command to launch the Streamlit app for both the server and client.
```

#### Sample Output:
```console
ubuntu@ip-10-0-0-137:~/serve$ ./examples/usecases/llm_diffusion_serving_app/docker/build_image.sh 
EXAMPLE_DIR: .//examples/usecases/llm_diffusion_serving_app/docker
ROOT_DIR: /home/ubuntu/serve
DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 --file .//examples/usecases/llm_diffusion_serving_app/docker/Dockerfile --build-arg BASE_IMAGE="pytorch/torchserve:latest-cpu" --build-arg EXAMPLE_DIR=".//examples/usecases/llm_diffusion_serving_app/docker" --build-arg HUGGINGFACE_TOKEN=hf_<token> --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg NO_PROXY= -t "pytorch/torchserve:llm_diffusion_serving_app" .
[+] Building 1.4s (18/18) FINISHED                                                                                                                                                               docker:default
 => [internal] load .dockerignore                                                                                                                                                                          0.0s
 => => transferring context: 2B                                                                                                                                                                            0.0s
 => [internal] load build definition from Dockerfile                                                                                                                                                       0.0s
 => => transferring dockerfile: 2.33kB                                                                                                                                                                     0.0s
 => [internal] load metadata for docker.io/pytorch/torchserve:latest-cpu                                                                                                                                   0.2s
 => [server  1/13] FROM docker.io/pytorch/torchserve:latest-cpu@sha256:50e189492f630a56214dce45ec6fd8db3ad45845890c0e8c26b469c6b06ca4fe                                                                    0.0s
 => [internal] load build context                                                                                                                                                                          0.0s
 => => transferring context: 3.54kB                                                                                                                                                                        0.0s
 => CACHED [server  2/13] RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt     apt-get update &&     apt-get install libopenmpi-dev git -y                                                          0.0s
 => CACHED [server  3/13] WORKDIR /home/model-server/                                                                                                                                                      0.0s
 => CACHED [server  4/13] COPY .//examples/usecases/llm_diffusion_serving_app/docker/requirements.txt /home/model-server/requirements.txt                                                                  0.0s
 => CACHED [server  5/13] RUN pip install -r requirements.txt                                                                                                                                              0.0s
 => CACHED [server  6/13] COPY .//examples/usecases/llm_diffusion_serving_app/docker/sd/requirements.txt /home/model-server/sd_requirements.txt                                                            0.0s
 => CACHED [server  7/13] RUN pip install -r sd_requirements.txt                                                                                                                                           0.0s
 => [server  8/13] COPY .//examples/usecases/llm_diffusion_serving_app/docker /home/model-server/llm_diffusion_serving_app/                                                                                0.0s
 => [server  9/13] RUN --mount=type=secret,id=hf_token     huggingface-cli login --token hf_<token>                                                                                      0.5s
 => [server 10/13] WORKDIR /home/model-server/llm_diffusion_serving_app                                                                                                                                    0.0s
 => [server 11/13] COPY .//examples/usecases/llm_diffusion_serving_app/docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh                                                                   0.0s 
 => [server 12/13] COPY .//examples/usecases/llm_diffusion_serving_app/docker/config.properties /home/model-server/config.properties                                                                       0.0s
 => [server 13/13] RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh     && chown -R model-server /home/model-server                                                                                       0.3s
 => exporting to image                                                                                                                                                                                     0.1s
 => => exporting layers                                                                                                                                                                                    0.1s
 => => writing image sha256:e900f12e6dad3ec443966766f82860427fa066aefe504a415eecf69bf4c3c043                                                                                                               0.0s
 => => naming to docker.io/pytorch/torchserve:llm_diffusion_serving_app                                                                                                                                    0.0s

Run the following command to start the Multi-image generation App

docker run --rm -it --platform linux/amd64 \
-p 127.0.0.1:8080:8080 \
-p 127.0.0.1:8081:8081 \
-p 127.0.0.1:8082:8082 \
-p 127.0.0.1:8084:8084 \
-p 127.0.0.1:8085:8085 \
-v /home/ubuntu/serve/model-store-local:/home/model-server/model-store \
-e MODEL_NAME_LLM=meta-llama/Llama-3.2-3B-Instruct \
-e MODEL_NAME_SD=stabilityai/stable-diffusion-xl-base-1.0  \
pytorch/torchserve:llm_diffusion_serving_app

Note: You can replace the model identifier as needed
```

## What to expect
Once you launch using the the docker run cmd, it launches two streamlit apps:
1. TorchServe Server App (running at http://localhost:8084) to start/stop TorchServe, load/register models, scale up/down workers. 
2. Client App (running at http://localhost:8085) where you can enter prompt for Image generation. 

#### Sample Output:

```console
ubuntu@ip-10-0-0-137:~/serve$ docker run --rm -it --platform linux/amd64 \
-p 127.0.0.1:8080:8080 \
-p 127.0.0.1:8081:8081 \
-p 127.0.0.1:8082:8082 \
-p 127.0.0.1:8084:8084 \
-p 127.0.0.1:8085:8085 \
-v /home/ubuntu/serve/model-store-local:/home/model-server/model-store \
-e MODEL_NAME_LLM=meta-llama/Llama-3.2-3B-Instruct \
-e MODEL_NAME_SD=stabilityai/stable-diffusion-xl-base-1.0  \
pytorch/torchserve:llm_diffusion_serving_app

Preparing meta-llama/Llama-3.2-1B-Instruct
/home/model-server/llm_diffusion_serving_app/llm /home/model-server/llm_diffusion_serving_app
Model meta-llama---Llama-3.2-1B-Instruct already downloaded.
Model archive for meta-llama---Llama-3.2-1B-Instruct exists.
/home/model-server/llm_diffusion_serving_app

Preparing stabilityai/stable-diffusion-xl-base-1.0
/home/model-server/llm_diffusion_serving_app/sd /home/model-server/llm_diffusion_serving_app
Model stabilityai/stable-diffusion-xl-base-1.0 already downloaded
Model archive for stabilityai---stable-diffusion-xl-base-1.0 exists.
/home/model-server/llm_diffusion_serving_app

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8085
  Network URL: http://123.11.0.2:8085
  External URL: http://123.123.12.34:8085


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8084
  Network URL: http://123.11.0.2:8084
  External URL: http://123.123.12.34:8084
```