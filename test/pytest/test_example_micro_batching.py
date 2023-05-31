import asyncio
import json
import random
import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZIP_STORED, ZipFile

import pytest
import requests
import test_utils
import yaml
from torchvision.models.resnet import ResNet18_Weights

from ts.torch_handler.unit_tests.test_utils.model_dir import download_model

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent

EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "microbatching")


def read_image_bytes(filename):
    with open(
        filename,
        "rb",
    ) as fin:
        image_bytes = fin.read()
    return image_bytes


@pytest.fixture(scope="module")
def kitten_image_bytes():
    return read_image_bytes(
        REPO_ROOT_DIR.joinpath(
            "examples/image_classifier/resnet_152_batch/images/kitten.jpg"
        ).as_posix()
    )


@pytest.fixture(scope="module")
def dog_image_bytes():
    return read_image_bytes(
        REPO_ROOT_DIR.joinpath(
            "examples/image_classifier/resnet_152_batch/images/dog.jpg"
        ).as_posix()
    )


@pytest.fixture(scope="module", params=[4, 16])
def mixed_batch(kitten_image_bytes, dog_image_bytes, request):
    batch_size = request.param
    labels = [
        "tiger_cat" if random.random() > 0.5 else "golden_retriever"
        for _ in range(batch_size)
    ]
    test_data = []
    for l in labels:
        test_data.append(kitten_image_bytes if l == "tiger_cat" else dog_image_bytes)
    return test_data, labels


@pytest.fixture(scope="module")
def model_name():
    yield "image_classifier"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module")
def serialized_file(work_dir):
    model_url = ResNet18_Weights.DEFAULT.url

    download_model(model_url, work_dir)

    yield Path(work_dir) / "model.pt"


@pytest.fixture(
    scope="module", name="mar_file_path", params=["yaml_config", "no_config"]
)
def create_mar_file(work_dir, serialized_file, model_archiver, model_name, request):
    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    name_file = REPO_ROOT_DIR.joinpath(
        "examples/image_classifier/resnet_18/index_to_name.json"
    ).as_posix()

    config_file = None
    if request.param == "yaml_config":
        micro_batching_params = {
            "micro_batching": {
                "micro_batch_size": 2,
                "parallelism": {
                    "preprocess": 2,
                    "inference": 2,
                    "postprocess": 2,
                },
            },
        }

        config_file = Path(work_dir).joinpath("model_config.yaml")

        with open(config_file, "w") as f:
            yaml.dump(micro_batching_params, f)
        config_file = REPO_ROOT_DIR.joinpath(
            "examples", "micro_batching", "config.yaml"
        )

    extra_files = [name_file]

    args = Namespace(
        model_name=model_name,
        version="1.0",
        serialized_file=str(serialized_file),
        model_file=REPO_ROOT_DIR.joinpath(
            "examples", "image_classifier", "resnet_18", "model.py"
        ).as_posix(),
        handler=REPO_ROOT_DIR.joinpath(
            "examples", "micro_batching", "micro_batching_handler.py"
        ).as_posix(),
        extra_files=",".join(extra_files),
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=config_file,
    )

    mock = MagicMock()
    mock.parse_args = MagicMock(return_value=args)
    with patch("archiver.ArgParser.export_model_args_parser", return_value=mock):
        # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
        with patch(
            "model_archiver.model_packaging_utils.zipfile.ZipFile",
            lambda x, y, _: ZipFile(x, y, ZIP_STORED),
        ):
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

    params = (
        ("model_name", model_name),
        ("url", file_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "32"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)


def test_single_example_inference(model_name, kitten_image_bytes):
    """
    Full circle test with torchserve
    """

    response = requests.post(
        url=f"http://localhost:8080/predictions/{model_name}", data=kitten_image_bytes
    )

    import inspect

    print(inspect.getmembers(response))

    assert response.status_code == 200


async def issue_request(model_name, data):
    return requests.post(
        url=f"http://localhost:8080/predictions/{model_name}", data=data
    )


async def issue_multi_requests(model_name, data):
    tasks = []
    for d in data:
        tasks.append(asyncio.create_task(issue_request(model_name, d)))

    ret = []
    for t in tasks:
        ret.append(await t)

    return ret


def test_multi_example_inference(model_name, mixed_batch):
    """
    Full circle test with torchserve
    """
    test_data, labels = mixed_batch

    responses = asyncio.run(issue_multi_requests(model_name, test_data))

    status_codes = [r.status_code for r in responses]

    assert status_codes == [200] * len(status_codes)

    result_entries = [json.loads(r.text) for r in responses]

    assert all(l in r.keys() for l, r in zip(labels, result_entries))
