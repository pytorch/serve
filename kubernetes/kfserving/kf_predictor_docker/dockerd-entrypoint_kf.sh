#!/bin/bash
set -e

    eval "$@"
    python /serve/kubernetes/kfserving/kfserving_wrapper/__main__.py 
    
# prevent docker exit
tail -f /dev/null
