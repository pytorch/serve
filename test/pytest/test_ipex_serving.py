import os
import sys
import json
from pathlib import Path
import subprocess
import yaml

import pytest
import requests
import test_utils
import torch
#from test_handler import run_inference_using_url_with_data

from unittest.mock import patch
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

from string import Template
import logging

from model_archiver.model_archiver_config import ModelArchiverConfig
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
snapshot_file_ipex = os.path.join(REPO_ROOT, "test/config_ipex.properties")
prompt_file = os.path.join(REPO_ROOT, "examples/large_models/ipex_llm_int8/sample_text_0.txt")

#CURR_FILE_PATH = Path(__file__).parent
HANDLER_PATH = os.path.join(REPO_ROOT, "examples/large_models/ipex_llm_int8/")
sys.path.append(HANDLER_PATH)


logger = logging.Logger(__name__)

PROMPTS = ["The capital of France is ",]

MANAGEMENT_API = "http://localhost:8081"
INFERENCE_API = "http://localhost:8080"


xeon_run_cpu_available = False

cmd = ["python", "-m", "torch.backends.xeon.run_cpu", "--no_python", "pwd"]
r = subprocess.run(cmd)
if r.returncode == 0:
    xeon_run_cpu_available = True

ipex_available = False
cmd = ["python", "-c", "import intel_extension_for_pytorch as ipex"]
r = subprocess.run(cmd)
if r.returncode == 0:
    ipex_available = True

ipex_xeon_run_available = xeon_run_cpu_available and ipex_available



# TODO: download each model we want to serve inside this folder (Change with model name)

# MODEL_FILE_PATH=HANDLER_PATH/"llama_2"



LLAMA_DEFAULT_CONFIG = f"""
    minWorkers: 1
    maxWorkers: 1
    responseTimeout: 1500
    batchSize: 4
    maxBatchDelay: 100
    
    handler:
        model_name: "meta-llama/Llama-2-7b-hf"
        clear_cache_dir: true
        quantized_model_path: "best_model.pt"
        example_inputs_mode: "MASK_KV_POS"
        to_channels_last: false
    
        # generation params
        batch_size: 1 # this batch size is mostly used for calibration, you can leave it as 1
        input_tokens: 1024
        max_new_tokens: 128
    
        # Use INT8 bf16 mix
        quant_with_amp: true
    
        # decoding technique
        greedy: true

    """

def test_handler_no_ipex(tmp_path, mocker):
    try:
        from llm_handler import IpexLLMHandler

        handler = IpexLLMHandler()
        ctx = MockContext()

        model_config_yaml = tmp_path/"model-config.yaml"
        #config = LLAMA_DEFAULT_CONFIG.substitute(
        #        {"nproc": "1", "stream": "true", "compile": compile, "ipex_enable":"false"}
        #)
        model_config_yaml.write_text(LLAMA_DEFAULT_CONFIG)
        os.environ["TS_IPEX_ENABLE"] = "false"

        with open(model_config_yaml, "r") as f:
            config = yaml.safe_load(f)

        ctx.model_yaml_config = config

        torch.manual_seed(42)
        handler.initialize(ctx)

        # The model with default ipex routine won't have "trace_graph" attribute
        assert hasattr(handler.user_model, "trace_graph") == False, "The default Pytorch module must not have 'trace_graph' attribute"

        x = handler.preprocess([{"data": json.dumps(PROMPTS[0])}])
        x = handler.inference(x)
        x = handler.postprocess(x)
        assert "Paris" in x[0], f"The Answer doesn't seem to be correct!"

    finally:
        del handler.user_model
        del handler

def test_handler_ipex_bf16(tmp_path, mocker):
    try:
        os.environ["TS_IPEX_ENABLE"] = "true"
        from llm_handler import IpexLLMHandler

        handler = IpexLLMHandler()
        ctx = MockContext()

        model_config_yaml = tmp_path/"model-config.yaml"
        #config = LLAMA_DEFAULT_CONFIG.substitute(
        #        {"nproc": "1", "stream": "true", "compile": compile, "ipex_enable":"false"}
        #)
        model_config_yaml.write_text(LLAMA_DEFAULT_CONFIG)

        with open(model_config_yaml, "r") as f:
            config = yaml.safe_load(f)

        ctx.model_yaml_config = config

        torch.manual_seed(42)
        handler.initialize(ctx)
        assert hasattr(handler.user_model, "trace_graph") == True, "IPEX optimized bf16 module must have 'trace_graph' attribute"

        x = handler.preprocess([{"data": json.dumps(PROMPTS[0])}])
        x = handler.inference(x)
        x = handler.postprocess(x)
        assert "Paris" in x[0], f"The Answer doesn't seem to be correct!"

    finally:
        del handler.user_model
        del handler
