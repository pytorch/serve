import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import test_utils
from model_archiver.model_archiver_config import ModelArchiverConfig

ACCELERATE_UNAVAILABLE = False
try:
    import accelerate  # nopycln: import
except ImportError:
    ACCELERATE_UNAVAILABLE = True

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
snapshot_file_ipex = os.path.join(REPO_ROOT, "test/config_ipex.properties")
default_ts_config = os.path.join(REPO_ROOT, "test/config_ts.properties")
prompt_file = os.path.join(
    REPO_ROOT, "examples/large_models/ipex_llm_int8/sample_text_0.txt"
)

# CURR_FILE_PATH = Path(__file__).parent
HANDLER_PATH = os.path.join(REPO_ROOT, "examples/large_models/ipex_llm_int8/")
sys.path.append(HANDLER_PATH)


logger = logging.Logger(__name__)

PROMPTS = [
    "The capital of France is ",
]

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


@pytest.fixture(scope="module")
def model_name():
    yield "llama2"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return Path(tmp_path_factory.mktemp(model_name))


# @pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name, model_config_yaml_file):
    mar_file_path = work_dir.joinpath(model_name + ".mar")

    handler_file = os.path.join(HANDLER_PATH, "llm_handler.py")
    assert Path(handler_file).exists()

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        serialized_file=None,
        model_file=None,
        handler=handler_file,
        extra_files=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=model_config_yaml_file.as_posix(),
    )

    with patch("archiver.ArgParser.export_model_args_parser", return_value=config):
        model_archiver.generate_model_archive()

        assert mar_file_path.exists()

        return mar_file_path.as_posix()


def run_inference_with_prompt(prompt_file, model_name):
    model_url = f"{INFERENCE_API}/predictions/{model_name}"
    response = run_inference_using_url_with_data(model_url, prompt_file)
    return response


def start_torchserve(ts_config_file):
    # start the torchserve
    test_utils.start_torchserve(
        model_store=test_utils.MODEL_STORE, snapshot_file=ts_config_file, gen_mar=False
    )


DEFAULT_CONFIG = f"""
    minWorkers: 1
    maxWorkers: 1
    responseTimeout: 1500
    batchSize: 4
    maxBatchDelay: 100

    handler:
        model_name: "baichuan-inc/Baichuan2-7B-Chat"
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

CONFIG_WOQ = f"""
    minWorkers: 1
    maxWorkers: 1
    responseTimeout: 1500
    batchSize: 4
    maxBatchDelay: 100

    handler:
        model_name: "baichuan-inc/Baichuan2-7B-Chat"
        clear_cache_dir: true
        quantized_model_path: "best_model.pt"
        example_inputs_mode: "MASK_KV_POS"
        to_channels_last: false

        # generation params
        batch_size: 1
        input_tokens: 1024
        max_new_tokens: 128

        # Use INT8 bf16 mix
        quant_with_amp: true

        # Woq params
        ipex_weight_only_quantization: true
        woq_dtype: "INT8"
        lowp_mode: "BF16"
        act_quant_mode: "PER_IC_BLOCK"
        group_size: -1

        # decoding technique
        greedy: true
    """

CONFIG_SQ = f"""
    minWorkers: 1
    maxWorkers: 1
    responseTimeout: 1500
    batchSize: 4
    maxBatchDelay: 100

    handler:
        model_name: "baichuan-inc/Baichuan2-7B-Chat"
        clear_cache_dir: true
        quantized_model_path: "best_model.pt"
        example_inputs_mode: "MASK_KV_POS"
        to_channels_last: false

        # generation params
        batch_size: 1
        input_tokens: 1024
        max_new_tokens: 128

        # use bf16-int8 mix
        quant_with_amp: true

        # SQ quantization params
        ipex_smooth_quantization: true
        calibration_dataset: "NeelNanda/pile-10k"
        calibration_split: "train"
        num_calibration_iters: 32
        alpha: 0.9

        # decoding technique
        greedy: true

    """


@pytest.mark.skipif(
    ACCELERATE_UNAVAILABLE, reason="HF accelerate library not available"
)
def test_handler_default_pytorch(work_dir, model_archiver):
    test_utils.torchserve_cleanup()
    # create_mar_file(work_dir, model_archiver, model_name, model_config_yaml_file):
    model_config_yaml = work_dir / "model-config.yaml"
    model_config_yaml.write_text(DEFAULT_CONFIG)

    # Create mar file
    model_name = "llama2_no_ipex"
    mar_file_path = create_mar_file(
        work_dir, model_archiver, model_name, model_config_yaml
    )
    os.makedirs(os.path.dirname(test_utils.MODEL_STORE), exist_ok=True)
    shutil.move(mar_file_path, test_utils.MODEL_STORE)

    # start torchserve server
    start_torchserve(default_ts_config)

    # load the model
    model_url = f"{MANAGEMENT_API}/models?url={model_name}.mar"
    requests.post(model_url)

    # query model info
    model_url = f"{MANAGEMENT_API}/models/{model_name}"
    response = requests.get(model_url)
    assert response.status_code == 200, "The default PyTorch Model failed to load"

    # send prompts to the model
    model_url = f"{INFERENCE_API}/predictions/{model_name}"
    response = requests.post(
        url=model_url,
        data=json.dumps(
            PROMPTS[0],
        ),
    )

    assert response.status_code == 200, "The model failed to generate text from prompt!"
    assert "Paris" in response.text, "The response doesn't seem to be correct!"

    test_utils.torchserve_cleanup()


@pytest.mark.skipif(
    ACCELERATE_UNAVAILABLE, reason="HF accelerate library not available"
)
def test_handler_ipex_bf16(work_dir, model_archiver):
    test_utils.torchserve_cleanup()
    # create_mar_file(work_dir, model_archiver, model_name, model_config_yaml_file):
    model_config_yaml = work_dir / "model-config.yaml"
    model_config_yaml.write_text(DEFAULT_CONFIG)

    # Create mar file
    model_name = "llama2_ipex_bf16"
    mar_file_path = create_mar_file(
        work_dir, model_archiver, model_name, model_config_yaml
    )
    os.makedirs(os.path.dirname(test_utils.MODEL_STORE), exist_ok=True)
    shutil.move(mar_file_path, test_utils.MODEL_STORE)

    # start torchserve server
    start_torchserve(snapshot_file_ipex)

    # load the model
    model_url = f"{MANAGEMENT_API}/models?url={model_name}.mar"
    requests.post(model_url)

    # query model info
    model_url = f"{MANAGEMENT_API}/models/{model_name}"
    response = requests.get(model_url)
    assert response.status_code == 200, "The IPEX bFloat16 model failed to initialize"

    # send prompts to the model
    model_url = f"{INFERENCE_API}/predictions/{model_name}"
    response = requests.post(
        url=model_url,
        data=json.dumps(
            PROMPTS[0],
        ),
    )

    assert response.status_code == 200, "The model failed to generate text from prompt!"
    assert "Paris" in response.text, "The response doesn't seem to be correct!"

    test_utils.torchserve_cleanup()


@pytest.mark.skipif(
    ACCELERATE_UNAVAILABLE, reason="HF accelerate library not available"
)
def test_handler_ipex_int8_woq(work_dir, model_archiver):
    test_utils.torchserve_cleanup()
    # create_mar_file(work_dir, model_archiver, model_name, model_config_yaml_file):
    model_config_yaml = work_dir / "model-config.yaml"
    model_config_yaml.write_text(CONFIG_WOQ)

    # Create mar file
    model_name = "llama2_ipex_int8_woq"
    mar_file_path = create_mar_file(
        work_dir, model_archiver, model_name, model_config_yaml
    )
    os.makedirs(os.path.dirname(test_utils.MODEL_STORE), exist_ok=True)
    shutil.move(mar_file_path, test_utils.MODEL_STORE)

    # start torchserve server
    start_torchserve(snapshot_file_ipex)

    # load the model
    model_url = f"{MANAGEMENT_API}/models?url={model_name}.mar"
    requests.post(model_url)

    # query model info
    model_url = f"{MANAGEMENT_API}/models/{model_name}"
    response = requests.get(model_url)
    assert (
        response.status_code == 200
    ), "The IPEX weight-only quantization Model failed to initialize"

    # send prompts to the model
    model_url = f"{INFERENCE_API}/predictions/{model_name}"
    response = requests.post(
        url=model_url,
        data=json.dumps(
            PROMPTS[0],
        ),
    )

    assert response.status_code == 200, "The model failed to generate text from prompt!"
    assert "Paris" in response.text, "The response doesn't seem to be correct!"

    test_utils.torchserve_cleanup()


@pytest.mark.skipif(
    ACCELERATE_UNAVAILABLE, reason="HF accelerate library not available"
)
def test_handler_ipex_int8_sq(work_dir, model_archiver):
    test_utils.torchserve_cleanup()
    # create_mar_file(work_dir, model_archiver, model_name, model_config_yaml_file):
    model_config_yaml = work_dir / "model-config.yaml"
    model_config_yaml.write_text(CONFIG_SQ)

    # Create mar file
    model_name = "llama2_ipex_int8_sq"
    mar_file_path = create_mar_file(
        work_dir, model_archiver, model_name, model_config_yaml
    )
    os.makedirs(os.path.dirname(test_utils.MODEL_STORE), exist_ok=True)
    shutil.move(mar_file_path, test_utils.MODEL_STORE)

    # start torchserve server
    start_torchserve(snapshot_file_ipex)

    # load the model
    model_url = f"{MANAGEMENT_API}/models?url={model_name}.mar"
    requests.post(model_url)

    # query model info
    model_url = f"{MANAGEMENT_API}/models/{model_name}"
    response = requests.get(model_url)
    assert (
        response.status_code == 200
    ), "The IPEX smoothquant quantized Model failed to load"

    # send prompts to the model
    model_url = f"{INFERENCE_API}/predictions/{model_name}"
    response = requests.post(
        url=model_url,
        data=json.dumps(
            PROMPTS[0],
        ),
    )

    assert response.status_code == 200, "The model failed to generate text from prompt!"
    assert "Paris" in response.text, "The response doesn't seem to be correct!"

    test_utils.torchserve_cleanup()
