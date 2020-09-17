#!/bin/bash
set -e

if [[ "$1" = *"model_dir"* ]] || [[ "$1" = *"model_name"* ]] || [[ "$1" = *"http_port"* ]] || [[ "$1" = *"grpc_port"* ]] || [[ "$1" = *"workers"* ]] || [[ "$1" = *"max_buffer_size"* ]]; then
    #starts torchserve 
    torchserve --start --ts-config /home/model-server/config.properties
    #Wrapper class starts kfserver 
    python /home/model-server/kfserving_wrapper/__main__.py $@

else
    eval "$@"
fi



# prevent docker exit
tail -f /dev/null