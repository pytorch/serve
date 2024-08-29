## Download repo and build Docker image

```console
export HUGGINGFACE_TOKEN=<HUGGINGFACE_TOKEN>

git clone -b gpt_fast_sd_ov https://github.com/likholat/serve.git
cd serve
./examples/LLM/llama/ov_chat_app/docker/build_image.sh
```

## Start the app

Use the command printed after successful docker build
