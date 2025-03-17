#!/bin/bash
set -e


if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --foreground --ts-config /home/model-server/config.properties --disable-token-auth "$@"
else
    eval "$@"

    # prevent docker exit
    tail -f /dev/null
fi
