import json
import shutil
import sys
from pathlib import Path
from string import Template

import pytest
import requests
import test_utils
from model_archiver.model_archiver_config import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
VLLM_PATH = CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "vllm"
LORA_SRC_PATH = VLLM_PATH / "lora"
CONFIG_PROPERTIES_PATH = CURR_FILE_PATH.parents[1] / "test" / "config_ts.properties"

LLAMA_MODEL_PATH = (
    LORA_SRC_PATH
    / "model"
    / "models--meta-llama--Llama-2-7b-hf"
    / "snapshots"
    / "01c7f73d771dfac7d292323805ebc428287df4f9"
)

ADAPTER_PATH = (
    LORA_SRC_PATH
    / "adapters"
    / "model"
    / "models--yard1--llama-2-7b-sql-lora-test"
    / "snapshots"
    / "0dfa347e8877a4d4ed19ee56c140fa518470028c"
)

YAML_CONFIG = Template(
    f"""
# TorchServe frontend parameters
minWorkers: 1
maxWorkers: 1
batchSize: 16
maxBatchDelay: 100
responseTimeout: 1200
deviceType: "gpu"
continuousBatching: true

handler:
    model_path: "{LLAMA_MODEL_PATH.as_posix()}"
    vllm_engine_config:
        enable_lora: true
        max_loras: 4
        max_cpu_loras: 4
        max_num_seqs: 256

    adapters:
        adapter_1: "{ADAPTER_PATH.as_posix()}"
"""
)

PROMPTS = [
    {
        "prompt": "A robot may not injure a human being",
        "max_new_tokens": 50,
        "temperature": 0.8,
        "logprobs": 1,
        "prompt_logprobs": 1,
        "max_tokens": 128,
        "adapter": "adapter_1",
    },
]

EXPECTED_RESULTS = [
    # ", Paris, is a city of romance, fashion, and art. The city is home to the Eiffel Tower, the Louvre, and the Arc de Triomphe. Paris is also known for its cafes, restaurants",
    " is Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is Madrid.\nThe capital of the United Kingdom is London.\nThe capital of the European Union is Brussels.\n",
    " is Paris.\n\nThe capital of Germany is Berlin.\n\nThe capital of Italy is Rome.\n\nThe capital of Spain is Madrid.\n\nThe capital of the United Kingdom is London.\n\nThe capital of the United States is",
]


def necessary_files_unavailable():
    return {
        "condition": not (LLAMA_MODEL_PATH.exists() and ADAPTER_PATH.exists()),
        "reason": f"Required files are not present (see README): {LLAMA_MODEL_PATH.as_posix()} + {ADAPTER_PATH.as_posix()}",
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
    yield "lora"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name, request):
    mar_file_path = Path(work_dir).joinpath(model_name)

    model_config_yaml = Path(work_dir) / "model-config.yaml"
    yaml_config = YAML_CONFIG.substitute(request.param)
    model_config_yaml.write_text(yaml_config)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        model_file=CURR_FILE_PATH.joinpath(
            "test_data", "streaming", "fake_streaming_model.py"
        ).as_posix(),
        handler=(VLLM_PATH / "base_vllm_handler.py").as_posix(),
        serialized_file=None,
        export_path=work_dir,
        requirements_file=(VLLM_PATH / "requirements.txt").as_posix(),
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


@pytest.fixture(scope="module", name="model_name_and_stdout")
def test_vllm_lora_mar(mar_file_path, model_store):
    """
    Register the model in torchserve
    """

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    shutil.copytree(mar_file_path, Path(model_store) / model_name)

    params = (
        ("model_name", model_name),
        ("url", Path(model_store) / model_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    test_utils.start_torchserve(
        model_store=model_store, snapshot_file=CONFIG_PROPERTIES_PATH, gen_mar=False
    )

    try:
        test_utils.reg_resp = test_utils.register_model_with_params(params)

        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            data=json.dumps(
                PROMPTS[0],
            ),
            stream=True,
        )

        assert response.status_code == 200

        assert response.headers["Transfer-Encoding"] == "chunked"

        prediction = []
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                data = json.loads(chunk)
                prediction += [data.get("text", "")]

        assert len(prediction) > 1

        assert "".join(prediction) == EXPECTED_RESULTS[1]
    finally:
        test_utils.unregister_model(model_name)

        # Clean up files
        shutil.rmtree(Path(model_store) / model_name)
