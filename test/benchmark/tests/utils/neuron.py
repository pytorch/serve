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

def setup_neuron_mar_files(connection=None, virtual_env_name=None, batch_size=1):
    activation_command = ""
    if virtual_env_name:
        activation_command = f"cd /home/ubuntu/serve/test/benchmark/tests/resources/neuron-bert && source activate {virtual_env_name} && "

    connection.run(f"{activation_command}python3 compile_bert.py --batch-size {batch_size}", warn=True)
    connection.run(f"sed -i 's/batch_size=[[:digit:]]\+/batch_size={batch_size}/g' config.py", warn=True)
    connection.run(f"mkdir -p /home/ubuntu/benchmark/model_store")
    connection.run(f"{activation_command}torch-model-archiver --model-name 'benchmark_{batch_size}' --version 1.0 --serialized-file ./bert_neuron_{batch_size}.pt --handler './handler_bert.py' --extra-files './config.py' --export-path /home/ubuntu/benchmark/model_store")
    pass