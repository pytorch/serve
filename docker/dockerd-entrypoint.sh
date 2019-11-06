#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    mxnet-model-server --start --mms-config config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
