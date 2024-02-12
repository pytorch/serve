import json
import shutil
import sys
from pathlib import Path
from string import Template
from unittest.mock import patch

import pytest
import requests
import test_utils
import torch
import yaml
from model_archiver.model_archiver_config import ModelArchiverConfig

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
GPT_PATH = CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "gpt_fast"
GPT_SRC_PATH = GPT_PATH / "gpt-fast"

LLAMA_MODEL_PATH = (
    GPT_SRC_PATH
    / "checkpoints"
    / "meta-llama"
    / "Llama-2-13b-chat-hf"
    / "model_int8.pth"
)

YAML_CONFIG = Template(
    f"""
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 200
responseTimeout: 300
parallelType: "tp"
deviceType: "gpu"
continuousBatching: false
torchrun:
    nproc-per-node: $nproc
handler:
    converted_ckpt_dir: "{LLAMA_MODEL_PATH.as_posix()}"
    max_new_tokens: 50
    compile: $compile
    stream: $stream
"""
)

MAR_PARAMS = (
    {
        "nproc": 1,
        "stream": "true",
        "compile": "true",
    },
    {
        "nproc": 4,
        "stream": "true",
        "compile": "false",
    },
    {
        "nproc": 4,
        "stream": f"true\n    speculate_k: 8\n    draft_checkpoint_path: '{(LLAMA_MODEL_PATH.parents[1] / 'Llama-2-7b-chat-hf' / 'model_int8.pth').as_posix()}'",
        "compile": "true",
    },
)

PROMPTS = [
    {
        "prompt": "The capital of France",
        "max_new_tokens": 50,
    },
]

EXPECTED_RESULTS = [
    # ", Paris, is a city of romance, fashion, and art. The city is home to the Eiffel Tower, the Louvre, and the Arc de Triomphe. Paris is also known for its cafes, restaurants",
    " is Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is Madrid.\nThe capital of the United Kingdom is London.\nThe capital of the European Union is Brussels.\n",
    " is Paris.\n\nThe capital of Germany is Berlin.\n\nThe capital of Italy is Rome.\n\nThe capital of Spain is Madrid.\n\nThe capital of the United Kingdom is London.\n\nThe capital of the United States is",
]


def necessary_files_unavailable():
    return {
        "condition": not (LLAMA_MODEL_PATH.exists() and GPT_SRC_PATH.exists()),
        "reason": f"Required files are not present (see README): {LLAMA_MODEL_PATH.as_posix()} + {GPT_SRC_PATH.as_posix()}",
    }


@pytest.fixture
def add_paths():
    sys.path.append(GPT_PATH.as_posix())
    sys.path.append(GPT_SRC_PATH.as_posix())
    yield
    sys.path.pop()
    sys.path.pop()


@pytest.mark.skipif(**necessary_files_unavailable())
@pytest.mark.parametrize(("compile"), ("false", "true"))
def test_handler(tmp_path, add_paths, compile, mocker):
    try:
        from handler import GptHandler

        handler = GptHandler()
        ctx = MockContext(
            model_pt_file=None,
            model_dir=GPT_PATH.as_posix(),
        )
        model_config_yaml = tmp_path / "model-config.yaml"
        config = YAML_CONFIG.substitute(
            {"nproc": "1", "stream": "true", "compile": compile}
        )
        model_config_yaml.write_text(config)

        with open(model_config_yaml, "r") as f:
            config = yaml.safe_load(f)

        ctx.model_yaml_config = config
        ctx.request_ids = {0: "0"}

        torch.manual_seed(42)
        handler.initialize(ctx)

        assert ("cuda:0" if torch.cuda.is_available() else "cpu") == str(handler.device)

        send_mock = mocker.MagicMock(name="send_intermediate_predict_response")
        with patch("handler.send_intermediate_predict_response", send_mock):
            x = handler.preprocess([{"data": json.dumps(PROMPTS[0])}])
            x = handler.inference(x)
            x = handler.postprocess(x)

        result = "".join(c[0][0][0] for c in send_mock.call_args_list)

        assert result == EXPECTED_RESULTS[1 if compile == "true" else 0]
    finally:
        # free memory in case of failed test
        del handler.model
        del handler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def model_name():
    yield "gpt_fast_handler"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path", params=MAR_PARAMS)
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
        handler=(GPT_PATH / "handler.py").as_posix(),
        serialized_file=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        config_file=model_config_yaml.as_posix(),
        extra_files=",".join(
            [
                (GPT_SRC_PATH / f).as_posix()
                for f in ("generate.py", "model.py", "quantize.py", "tp.py")
            ]
        ),
        archive_format="no-archive",
    )

    model_archiver.generate_model_archive(config)

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    # Clean up files
    shutil.rmtree(mar_file_path)


@pytest.fixture(scope="module", name="model_name_and_stdout")
def register_model(mar_file_path, model_store, torchserve):
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

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name, torchserve

    test_utils.unregister_model(model_name)

    # Clean up files
    shutil.rmtree(Path(model_store) / model_name)


@pytest.mark.skipif(**necessary_files_unavailable())
def test_gpt_fast_mar(model_name_and_stdout):
    model_name, _ = model_name_and_stdout

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
            prediction += [chunk.decode("utf-8")]

    assert len(prediction) > 1

    assert "".join(prediction) == EXPECTED_RESULTS[1]
