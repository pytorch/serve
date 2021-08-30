#!/bin/bash
set -e

    eval "$@"
    python /home/model-server/kfserving_wrapper/__main__.py 
    
# prevent docker exit
tail -f /dev/null
