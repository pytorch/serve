#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /home/model-server/config.properties
    python /home/model-server/kfserving_wrapper/__main__.py
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
