#!/bin/bash

CONTAINER=pytorch/torchserve:0.8.2-cpu

docker run --rm \
-v $PWD:/home/model-server \
--entrypoint /bin/bash \
--workdir /home/model-server \
$CONTAINER \
-c \
"pip install -r requirements.txt && python save_jit_model.py
"