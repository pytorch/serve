import importlib
import json
import os
import shutil
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
import requests
import test_utils
import torch
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "torchrec_dlrm")

EXPECTED_RESULT_BS_2 = [
    {"default": [pytest.approx(0.1051536425948143), pytest.approx(0.10522478073835373)]}
]
EXPECTED_RESULT_BS_1 = [{"default": [pytest.approx(0.1051536425948143)]}]

TEST_CASES = [
    ("dlrm_bs_1.json", EXPECTED_RESULT_BS_1),
    ("dlrm_bs_2.json", EXPECTED_RESULT_BS_2),
]


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("work_dir")


def simple_dlrm_factory():
    @dataclass
    class DLRMModelConfig:
        dense_arch_layer_sizes: List[int]
        dense_in_features: int
        embedding_dim: int
        id_list_features_keys: List[str]
        num_embeddings_per_feature: List[int]
        over_arch_layer_sizes: List[int]

    return DLRMModelConfig(
        dense_arch_layer_sizes=[32, 16, 8],
        dense_in_features=len(DEFAULT_INT_NAMES),
        embedding_dim=8,
        id_list_features_keys=DEFAULT_CAT_NAMES,
        num_embeddings_per_feature=len(DEFAULT_CAT_NAMES)
        * [
            3,
        ],
        over_arch_layer_sizes=[32, 32, 16, 1],
    )


@pytest.fixture(scope="module", name="serialized_file")
def create_serialized_file(work_dir, session_mocker):
    script_path = EXAMPLE_ROOT_DIR / "create_dlrm_mar.py"

    loader = importlib.machinery.SourceFileLoader(
        "create_dlrm_mar", script_path.as_posix()
    )
    spec = importlib.util.spec_from_loader("create_dlrm_mar", loader)
    create_dlrm_mar = importlib.util.module_from_spec(spec)

    sys.modules["create_dlrm_mar"] = create_dlrm_mar

    loader.exec_module(create_dlrm_mar)

    session_mocker.patch(
        "dlrm_factory.create_default_model_config", simple_dlrm_factory
    )

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

    mar_file_path = os.path.join(work_dir, model_name + ".mar")

    args = Namespace(
        model_name=model_name,
        version="1.0",
        serialized_file=str(serialized_file),
        model_file=CURR_FILE_PATH.joinpath("test_data", "dlrm_model.py").as_posix(),
        handler=EXAMPLE_ROOT_DIR.joinpath("dlrm_handler.py").as_posix(),
        extra_files=EXAMPLE_ROOT_DIR.joinpath("dlrm_factory.py").as_posix(),
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
    )

    print(args)

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

    assert os.path.exists(mar_file_path)

    yield mar_file_path

    # Clean up files
    try:
        os.remove(mar_file_path)
    except OSError:
        pass


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    shutil.copy(mar_file_path, model_store)

    file_name = os.path.split(mar_file_path)[-1]

    model_name = os.path.splitext(file_name)[0]

    test_utils.reg_resp = test_utils.register_model(model_name, file_name)

    yield model_name

    test_utils.unregister_model(model_name)


@pytest.mark.parametrize(("file", "expected_result"), TEST_CASES)
def test_handler(monkeypatch, mocker, file, expected_result):
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)
    mocker.patch("dlrm_factory.create_default_model_config", simple_dlrm_factory)

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
def test_inference_with_untrained_model(model_name, file, expected_result):

    with open(Path(CURR_FILE_PATH) / "test_data" / file, "rb") as f:
        response = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}", data=f
        )

    assert response.status_code == 200

    result_entries = json.loads(response.text)

    assert [result_entries] == expected_result
