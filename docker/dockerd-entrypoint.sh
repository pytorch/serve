#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
