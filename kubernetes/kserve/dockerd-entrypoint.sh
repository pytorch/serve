#!/bin/bash
set -e

    eval "$@"
    python /home/model-server/kserve_wrapper/__main__.py 
    
# prevent docker exit
tail -f /dev/null
