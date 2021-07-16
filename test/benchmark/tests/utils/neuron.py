import subprocess
import time
import glob
import os
import requests
import tempfile

import invoke
import pandas as pd

from io import StringIO
from urllib.parse import urlparse
from invoke import run
from invoke.context import Context

from . import DEFAULT_REGION, IAM_INSTANCE_PROFILE, AMI_ID, LOGGER, S3_BUCKET_BENCHMARK_ARTIFACTS

# Assumes the functions from this file execute on an Ubuntu ec2 instance
ROOT_DIR = f"/home/ubuntu"
TORCHSERVE_DIR = os.path.join(ROOT_DIR, "serve")
MODEL_STORE = os.path.join(TORCHSERVE_DIR, "model_store")
LOCAL_TMP_DIR = "/tmp"
TMP_DIR = "/home/ubuntu"
NEURON_RESOURCES_FOLDER = os.path.join(TORCHSERVE_DIR, "test", "benchmark", "tests", "resources", "neuron-bert")

def setup_neuron_mar_files(connection=None, virtual_env_name=None, batch_size=1):
    activation_command = ""
    
    if virtual_env_name:
        activation_command = f"cd /home/ubuntu/serve/test/benchmark/tests/resources/neuron-bert && source activate {virtual_env_name} && "
    
    # Note: change version here to make sure the torch version compatible with neuron is being used.
    connection.run(f"{activation_command}pip3 install -U --ignore-installed torch==1.7.1", warn=True)
    connection.run(f"{activation_command}pip3 install -U --ignore-installed torch-neuron 'neuron-cc[tensorflow]' --extra-index-url=https://pip.repos.neuron.amazonaws.com", warn=True)
    
    connection.run(f"{activation_command}python3 compile_bert.py --batch-size {batch_size}", warn=True)
    time.sleep(5)
    run_out_sed = connection.run(f"{activation_command}sed -i 's/batch_size=[[:digit:]]\+/batch_size={batch_size}/g' config.py", warn=True)
    LOGGER.info(f"run_out_sed: {run_out_sed.stdout}, run_out_return: {run_out_sed.return_code}")
    run_out_mkdir = connection.run(f"mkdir -p /home/ubuntu/benchmark/model_store")
    LOGGER.info(f"run_out_mkdir: {run_out_mkdir.stdout}, run_out_return: {run_out_mkdir.return_code}")
    run_out_archiver = connection.run(f"{activation_command}torch-model-archiver --model-name 'benchmark_{batch_size}' --version 1.0 --serialized-file ./bert_neuron_{batch_size}.pt --handler './handler_bert.py' --extra-files './config.py' -f", warn=True)
    LOGGER.info(f"run_out_archiver: {run_out_archiver.stdout}, run_out_return: {run_out_archiver.return_code}")
    
    LOGGER.info(f"Running copy command")
    connection.run(f"cp /home/ubuntu/serve/test/benchmark/tests/resources/neuron-bert/benchmark_{batch_size}.mar /home/ubuntu/benchmark/model_store")
    run_out = connection.run(f"test -e /home/ubuntu/benchmark/model_store/benchmark_{batch_size}.mar")
    if run_out.return_code == 0:
        LOGGER.info(f"mar file available at location /home/ubuntu/benchmark/model_store/benchmark_{batch_size}.mar")
    else:
        LOGGER.info(f"mar file NOT available at location /home/ubuntu/benchmark/model_store/benchmark_{batch_size}.mar")

    time.sleep(5)