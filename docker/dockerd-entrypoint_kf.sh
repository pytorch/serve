#!/bin/bash
set -e

#starts torchserve 
torchserve --start --ts-config /home/model-server/config.properties
#Wrapper class starts kfserver 
python /home/model-server/kfserving_wrapper/__main__.py

# prevent docker exit
tail -f /dev/null