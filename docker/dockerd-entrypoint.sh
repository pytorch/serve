#!/bin/bash
set -e

shift 1
torchserve --start --ts-config /home/model-server/config.properties
# prevent docker exit
tail -f /dev/null