"""
Unit test for the TorchRec DLRM example
"""
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import requests
import test_utils
import torch

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "torchrec_dlrm")
SIMPLE_MODEL_FACTORY_PY = CURR_FILE_PATH.joinpath("test_data", "dlrm_model.py")


# The test cases and expected results used for the unittests with batch size one and two.
EXPECTED_RESULTS = [
    [{"default": [pytest.approx(0.1051536425948143)]}],
    [
        {
            "default": [
                pytest.approx(0.1051536425948143),
                pytest.approx(0.10522478073835373),
            ]
        }
    ],
]
TEST_CASES = [
    ("dlrm_bs_1.json", EXPECTED_RESULTS[0]),
    ("dlrm_bs_2.json", EXPECTED_RESULTS[1]),
]


pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available())
    or (tuple(map(int, torch.version.cuda.split("."))) < (11, 3)),
    reason="CUDA is not available or CUDA version is <11.3",
)


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("work_dir")


@pytest.fixture(scope="module", name="model_config")
def get_simple_dlrm_model_config(monkeysession):
    """
    Helper fixture to create a simpler DLRM model which is also used in the MAR file
    """
    monkeysession.syspath_prepend(EXAMPLE_ROOT_DIR)
    simple_model_factory = test_utils.load_module_from_py_file(
        SIMPLE_MODEL_FACTORY_PY.as_posix()
    )
    yield simple_model_factory.simple_dlrm_model_config


@pytest.fixture(scope="module", name="serialized_file")
def create_serialized_file(work_dir, session_mocker, model_config):
    """
    This fixture creates the the simplified DLRM model and saves its satte_dict to disk.
    """
    script_path = EXAMPLE_ROOT_DIR / "create_dlrm_mar.py"
    create_dlrm_mar = test_utils.load_module_from_py_file(script_path.as_posix())
    sys.modules["create_dlrm_mar"] = create_dlrm_mar

    session_mocker.patch("dlrm_factory.create_default_model_config", model_config)

    MODEL_PT_FILE = work_dir / "dlrm.pt"

    torch.manual_seed(42 * 42)
    create_dlrm_mar.create_pt_file(MODEL_PT_FILE)

    return MODEL_PT_FILE


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, session_mocker, serialized_file, model_archiver):
    """
    Create mar file and return file path.
    """
    model_name = "scriptable_tokenizer_untrained"

    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    args = Namespace(
        model_name=model_name,
        version="1.0",
        serialized_file=str(serialized_file),
        model_file=SIMPLE_MODEL_FACTORY_PY.as_posix(),
        handler=EXAMPLE_ROOT_DIR.joinpath("dlrm_handler.py").as_posix(),
        extra_files=EXAMPLE_ROOT_DIR.joinpath("dlrm_factory.py").as_posix()
        + ","
        + EXAMPLE_ROOT_DIR.joinpath("dlrm_model_config.py").as_posix(),
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
    )

    mock = session_mocker.MagicMock()
    mock.parse_args = session_mocker.MagicMock(return_value=args)
    session_mocker.patch(
        "archiver.ArgParser.export_model_args_parser", return_value=mock
    )

    # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
    from zipfile import ZIP_STORED, ZipFile

    session_mocker.patch(
        "model_archiver.model_packaging_utils.zipfile.ZipFile",
        lambda x, y, _: ZipFile(x, y, ZIP_STORED),
    )

    model_archiver.generate_model_archive()

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    # Clean up files
    mar_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    """
    Register the model in torchserve
    """
    shutil.copy(mar_file_path, model_store)

    file_name = Path(mar_file_path).name

    model_name = Path(file_name).stem

    test_utils.reg_resp = test_utils.register_model(model_name, file_name)

    yield model_name

    test_utils.unregister_model(model_name)


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES)
def test_handler(monkeypatch, mocker, file, expected_result, model_config):
    """
    Test dlrm handler as standalone entity with specified test cases
    """
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)
    mocker.patch("dlrm_factory.create_default_model_config", model_config)

    from dlrm_handler import TorchRecDLRMHandler

    handler = TorchRecDLRMHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=EXAMPLE_ROOT_DIR.as_posix(),
        model_file="dlrm_factory.py",
    )

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)

    # Batch size 2
    with open(Path(CURR_FILE_PATH) / "test_data" / file) as f:
        data = json.load(f)

    x = mocker.Mock(get=lambda x: json.dumps(data))

    x = handler.preprocess([x])
    x = handler.inference(x)
    x = handler.postprocess(x)

    assert x == expected_result


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES)
def test_inference_with_untrained_model_post_as_text(model_name, file, expected_result):
    """
    Full circle test with torchserve
    """

    with open(Path(CURR_FILE_PATH) / "test_data" / file, "rb") as f:
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}", data=f
        )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert [result_entries] == expected_result


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES)
def test_inference_with_untrained_model_post_as_json(model_name, file, expected_result):
    """
    Full circle test with torchserve
    """

    with open(Path(CURR_FILE_PATH) / "test_data" / file, "rb") as f:
        json_data = json.load(f)
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}", json=json_data
        )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert [result_entries] == expected_result
