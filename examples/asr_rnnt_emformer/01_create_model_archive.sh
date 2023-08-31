#!/bin/bash

CONTAINER=pytorch/torchserve:0.8.2-cpu
# CONTAINER=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-cpu-py38-ubuntu20.04-e3

# create mar
docker run --rm \
-v $PWD:/home/model-server \
--entrypoint /bin/bash \
--workdir /home/model-server \
$CONTAINER \
-c \
"torch-model-archiver \
--model-name rnnt \
--version 1.0 \
--serialized-file decoder_jit.pt \
--handler handler.py \
--extra-files 1089-134686.trans.txt \
--requirements-file requirements.txt \
--force \
&& mkdir -p model-store \
&& mv rnnt.mar model-store/
"

# serve; /home/model-server/config.properties has pre-defined model-store location
docker run --rm --network host \
-p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
-v $PWD:/home/model-server \
$CONTAINER