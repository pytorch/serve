import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import requests
import test_utils
import torch
from model_archiver.model_archiver_config import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
VLLM_PATH = CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "vllm"
LORA_SRC_PATH = VLLM_PATH / "lora"
CONFIG_PROPERTIES_PATH = CURR_FILE_PATH.parents[1] / "test" / "config_ts.properties"

LLAMA_MODEL_PATH = "model/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"

ADAPTER_PATH = "adapters/model/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c"

YAML_CONFIG = f"""
# TorchServe frontend parameters
minWorkers: 1
maxWorkers: 1
batchSize: 1
maxBatchDelay: 100
responseTimeout: 1200
deviceType: "gpu"
asyncCommunication: true
parallelType: "custom"
parallelLevel: {torch.cuda.device_count()}

handler:
    model_path: "{(LORA_SRC_PATH / LLAMA_MODEL_PATH).as_posix()}"
    vllm_engine_config:
        enable_lora: true
        max_loras: 4
        max_cpu_loras: 4
        max_num_seqs: 16
        max_model_len: 250
        tensor_parallel_size: {torch.cuda.device_count()}
        served_model_name:
            - "Llama-2-7b-chat-hf"

    adapters:
        adapter_1: "{(LORA_SRC_PATH / ADAPTER_PATH).as_posix()}"
"""

PROMPTS = [
    {
        "prompt": "A robot may not injure a human being",
        "temperature": 0.8,
        "logprobs": 1,
        # "prompt_logprobs": 1,
        "max_tokens": 128,
        "model": "Llama-2-7b-chat-hf",
        # "model": "adapter_1",
        "stream": True,
    },
    {
        "prompt": "Paris is, ",
        "logprobs": 1,
        # "prompt_logprobs": 1,
        "max_tokens": 128,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 0,
        "model": "Llama-2-7b-chat-hf",
        "model": "adapter_1",
        "seed": 42,
        "stream": True,
    },
]
EXPECTED = [
    " or, ",  # through inaction", # edit to pass see https://github.com/vllm-project/vllm/issues/5404
    "1900.\n\nThe city is",  # bathed",
]

try:
    import vllm  # noqa

    VLLM_MISSING = False
except ImportError:
    VLLM_MISSING = True

try:
    from openai import OpenAI  # noqa

    OPENAI_MISSING = False
except ImportError:
    OPENAI_MISSING = True


def necessary_files_unavailable(profile=None):
    LLAMA = LORA_SRC_PATH / LLAMA_MODEL_PATH
    ADAPTER = LORA_SRC_PATH / ADAPTER_PATH
    if not (LLAMA.exists() and ADAPTER.exists()):
        return {
            "condition": True,
            "reason": f"Required files are not present (see README): {LLAMA.as_posix()} + {ADAPTER.as_posix()}",
        }
    elif VLLM_MISSING:
        return {
            "condition": True,
            "reason": f"VLLM is not installed",
        }
    elif profile == "openai" and OPENAI_MISSING:
        return {
            "condition": True,
            "reason": f"OpenAI client is not installed",
        }
    else:
        return {
            "condition": False,
            "reason": "None",
        }


@pytest.fixture
def add_paths():
    sys.path.append(VLLM_PATH.as_posix())
    sys.path.append(LORA_SRC_PATH.as_posix())
    yield
    sys.path.pop()
    sys.path.pop()


@pytest.fixture(scope="module")
def model_name():
    yield "Llama-2-7b-chat-hf"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name, request):
    mar_file_path = Path(work_dir).joinpath(model_name)

    model_config_yaml = Path(work_dir) / "model-config.yaml"
    model_config_yaml.write_text(YAML_CONFIG)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        handler="vllm_handler",
        serialized_file=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        config_file=model_config_yaml.as_posix(),
        archive_format="no-archive",
    )

    model_archiver.generate_model_archive(config)

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    # Clean up files
    shutil.rmtree(mar_file_path)


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    """
    Register the model in torchserve
    """
    file_name = Path(mar_file_path).name
    model_name = Path(file_name).stem

    shutil.copytree(mar_file_path, model_store + f"/{model_name}")

    params = (
        ("model_name", model_name),
        ("url", file_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)

    # Clean up files
    shutil.rmtree(Path(model_store) / model_name)


@pytest.mark.skipif(**necessary_files_unavailable())
def test_vllm_lora_mar(model_name):
    """
    Register the model in torchserve
    """

    responses = []

    for _ in range(10):
        idx = random.randint(0, 1)
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            json=PROMPTS[idx],
            stream=True,
        )

        assert response.status_code == 200

        assert response.headers["Transfer-Encoding"] == "chunked"
        responses += [(response, EXPECTED[idx])]

    predictions = []
    expected_result = []
    for response, expected in responses:
        prediction = []
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                data = json.loads(chunk)
                prediction += [data.get("text", "")]
        predictions += [prediction]
        expected_result += [expected]
    assert all(len(p) > 1 for p in predictions)
    assert all("".join(p).startswith(e) for p, e in zip(predictions, expected_result))


@pytest.mark.skipif(**necessary_files_unavailable("openai"))
@pytest.mark.parametrize("stream", [True, False])
def test_openai_api(model_name, stream):
    from openai import OpenAI, Stream
    from openai.types.completion import Completion

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:8080/predictions/{model_name}/1.0/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    response = client.completions.create(
        model=model_name, prompt="Hello world", temperature=0.0, stream=stream
    )

    if stream:
        assert isinstance(response, Stream)

        assert response.response.headers["Transfer-Encoding"] == "chunked"

        EXPECTED = "! I'm a new member of the community and I'm excited to"
        i = 0

        for chunk in response:
            assert isinstance(chunk, Completion)
            text = chunk.choices[0].text
            assert text == EXPECTED[i : i + len(text)]
            i += len(text)

    else:
        assert isinstance(response, Completion)

        assert (
            response.choices[0].text
            == "! I'm a new member of the community and I'm excited to"
        )
