import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import packaging.version
import pytest
import torch
from test_data.torch_compile.compile_handler import CompileHandler

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

PT_2_AVAILABLE = (
    True
    if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0")
    else False
)


CURR_FILE_PATH = Path(__file__).parent
TEST_DATA_DIR = os.path.join(CURR_FILE_PATH, "test_data", "torch_compile")

MODEL = "model.py"
MODEL_FILE = os.path.join(TEST_DATA_DIR, MODEL)
HANDLER_FILE = os.path.join(TEST_DATA_DIR, "compile_handler.py")
YAML_CONFIG_STR = os.path.join(TEST_DATA_DIR, "pt2.yaml")  # backend as string
YAML_CONFIG_DICT = os.path.join(TEST_DATA_DIR, "pt2_dict.yaml")  # arbitrary kwargs dict
YAML_CONFIG_ENABLE = os.path.join(
    TEST_DATA_DIR, "pt2_enable_true.yaml"
)  # arbitrary kwargs dict
YAML_CONFIG_ENABLE_FALSE = os.path.join(
    TEST_DATA_DIR, "pt2_enable_false.yaml"
)  # arbitrary kwargs dict
YAML_CONFIG_ENABLE_DEFAULT = os.path.join(
    TEST_DATA_DIR, "pt2_enable_default.yaml"
)  # arbitrary kwargs dict


SERIALIZED_FILE = os.path.join(TEST_DATA_DIR, "model.pt")
MODEL_STORE_DIR = os.path.join(TEST_DATA_DIR, "model_store")
MODEL_NAME = "half_plus_two"
EXPECTED_RESULT = 3.5


@pytest.fixture(scope="function")
def chdir_example(monkeypatch):
    # Change directory to example directory
    monkeypatch.chdir(TEST_DATA_DIR)
    monkeypatch.syspath_prepend(TEST_DATA_DIR)
    yield

    # Teardown
    monkeypatch.undo()

    # Delete imported model
    model = MODEL.split(".")[0]
    if model in sys.modules:
        del sys.modules[model]


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Skipping test on non-Linux system"
)
@pytest.mark.skipif(PT_2_AVAILABLE == False, reason="torch version is < 2.0.0")
class TestTorchCompile:
    def teardown_class(self):
        subprocess.run("torchserve --stop", shell=True, check=True)
        time.sleep(10)

    def test_archive_model_artifacts(self):
        assert len(glob.glob(MODEL_FILE)) == 1
        assert len(glob.glob(YAML_CONFIG_STR)) == 1
        assert len(glob.glob(YAML_CONFIG_DICT)) == 1
        subprocess.run(f"cd {TEST_DATA_DIR} && python model.py", shell=True, check=True)
        subprocess.run(f"mkdir -p {MODEL_STORE_DIR}", shell=True, check=True)

        # register 2 models, one with the backend as str config, the other with the kwargs as dict config
        subprocess.run(
            f"torch-model-archiver --model-name {MODEL_NAME}_str --version 1.0 --model-file {MODEL_FILE} --serialized-file {SERIALIZED_FILE} --config-file {YAML_CONFIG_STR} --export-path {MODEL_STORE_DIR} --handler {HANDLER_FILE} -f",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"torch-model-archiver --model-name {MODEL_NAME}_dict --version 1.0 --model-file {MODEL_FILE} --serialized-file {SERIALIZED_FILE} --config-file {YAML_CONFIG_DICT} --export-path {MODEL_STORE_DIR} --handler {HANDLER_FILE} -f",
            shell=True,
            check=True,
        )
        assert len(glob.glob(SERIALIZED_FILE)) == 1
        assert (
            len(glob.glob(os.path.join(MODEL_STORE_DIR, f"{MODEL_NAME}_str.mar"))) == 1
        )
        assert (
            len(glob.glob(os.path.join(MODEL_STORE_DIR, f"{MODEL_NAME}_dict.mar"))) == 1
        )

    def test_start_torchserve(self):
        cmd = f"torchserve --start --ncs --models {MODEL_NAME}_str.mar,{MODEL_NAME}_dict.mar --model-store {MODEL_STORE_DIR} --enable-model-api --disable-token-auth"
        subprocess.run(
            cmd,
            shell=True,
            check=True,
        )
        time.sleep(10)
        assert len(glob.glob("logs/access_log.log")) == 1
        assert len(glob.glob("logs/model_log.log")) == 1
        assert len(glob.glob("logs/ts_log.log")) == 1

    @pytest.mark.skipif(
        os.environ.get("TS_RUN_IN_DOCKER", False),
        reason="Test to be run outside docker",
    )
    def test_server_status(self):
        result = subprocess.run(
            "curl http://localhost:8080/ping",
            shell=True,
            capture_output=True,
            check=True,
        )
        expected_server_status_str = '{"status": "Healthy"}'
        expected_server_status = json.loads(expected_server_status_str)
        assert json.loads(result.stdout) == expected_server_status

    @pytest.mark.skipif(
        os.environ.get("TS_RUN_IN_DOCKER", False),
        reason="Test to be run outside docker",
    )
    def test_registered_model(self):
        result = subprocess.run(
            "curl http://localhost:8081/models",
            shell=True,
            capture_output=True,
            check=True,
        )

        def _response_to_tuples(response_str):
            models = json.loads(response_str)["models"]
            return {(k, v) for d in models for k, v in d.items()}

        # transform to set of tuples so order won't cause inequality
        expected_registered_model_str = '{"models": [{"modelName": "half_plus_two_str", "modelUrl": "half_plus_two_str.mar"}, {"modelName": "half_plus_two_dict", "modelUrl": "half_plus_two_dict.mar"}]}'
        assert _response_to_tuples(result.stdout) == _response_to_tuples(
            expected_registered_model_str
        )

    @pytest.mark.skipif(
        os.environ.get("TS_RUN_IN_DOCKER", False),
        reason="Test to be run outside docker",
    )
    def test_serve_inference(self):
        request_data = {"instances": [[1.0], [2.0], [3.0]]}
        request_json = json.dumps(request_data)

        for model_name in [f"{MODEL_NAME}_str", f"{MODEL_NAME}_dict"]:
            result = subprocess.run(
                f"curl -s -X POST -H \"Content-Type: application/json;\" http://localhost:8080/predictions/{model_name} -d '{request_json}'",
                shell=True,
                capture_output=True,
                check=True,
            )

            string_result = result.stdout.decode("utf-8")
            float_result = float(string_result)
            expected_result = 3.5

            assert float_result == expected_result

        model_log_path = glob.glob("logs/model_log.log")[0]
        with open(model_log_path, "rt") as model_log_file:
            model_log = model_log_file.read()
            assert "Compiled model with backend inductor\n" in model_log
            assert (
                "Compiled model with backend inductor, mode reduce-overhead"
                in model_log
            )

    @pytest.mark.parametrize(
        ("compile"), ("disabled", "enabled", "enabled_reduce_overhead")
    )
    def test_compile_inference_enable_options(self, chdir_example, compile):
        # Reset dynamo
        torch._dynamo.reset()

        # Handler
        handler = CompileHandler()

        if compile == "enabled":
            model_yaml_config_file = YAML_CONFIG_ENABLE_DEFAULT
        elif compile == "disabled":
            model_yaml_config_file = YAML_CONFIG_ENABLE_FALSE
        elif compile == "enabled_reduce_overhead":
            model_yaml_config_file = YAML_CONFIG_ENABLE

        # Context definition
        ctx = MockContext(
            model_pt_file=SERIALIZED_FILE,
            model_dir=TEST_DATA_DIR,
            model_file=MODEL,
            model_yaml_config_file=model_yaml_config_file,
        )

        torch.manual_seed(42 * 42)
        handler.initialize(ctx)
        handler.context = ctx

        # Check that model is compiled using dynamo
        if compile == "enabled" or compile == "enabled_reduce_overhead":
            assert isinstance(handler.model, torch._dynamo.OptimizedModule)
        else:
            assert not isinstance(handler.model, torch._dynamo.OptimizedModule)

        # Data for testing
        data = {"body": {"instances": [[1.0], [2.0], [3.0]]}}

        result = handler.handle([data], ctx)

        assert result[0] == EXPECTED_RESULT
