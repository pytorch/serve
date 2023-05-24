"""
Unit test for MicroBatchHandler class.
"""
import json
import random
import sys
from pathlib import Path

import pytest
from torchvision.models.resnet import ResNet18_Weights

from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
from ts.torch_handler.unit_tests.test_utils.model_dir import copy_files, download_model

REPO_DIR = Path(__file__).parents[3]


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
        REPO_DIR.joinpath(
            "examples/image_classifier/resnet_152_batch/images/kitten.jpg"
        ).as_posix()
    )


@pytest.fixture(scope="module")
def dog_image_bytes():
    return read_image_bytes(
        REPO_DIR.joinpath(
            "examples/image_classifier/resnet_152_batch/images/dog.jpg"
        ).as_posix()
    )


@pytest.fixture(scope="module")
def model_name():
    return "image_classifier"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, model_name):
    model_dir = tmp_path_factory.mktemp("image_classifier_model_dir")

    src_dir = REPO_DIR.joinpath("examples/image_classifier/resnet_18/")

    model_url = ResNet18_Weights.DEFAULT.url

    download_model(model_url, model_dir)

    files = {
        "model.py": model_name + ".py",
        "index_to_name.json": "index_to_name.json",
    }

    copy_files(src_dir, model_dir, files)

    sys.path.append(model_dir.as_posix())
    yield model_dir
    sys.path.pop()


@pytest.fixture(scope="module")
def context(model_dir, model_name):
    micro_batching_params = {
        "mb_size": 2,
        "mb_parallelism": {
            "preprocess": 1,
            "inference": 2,
            "postprocess": 3,
        },
    }

    config_file = Path(model_dir).joinpath("micro_batching.json")

    with open(config_file, "w") as f:
        json.dump(micro_batching_params, f)

    context = MockContext(
        model_name="mnist",
        model_dir=model_dir.as_posix(),
        model_file=model_name + ".py",
    )
    context.model_yaml_config = micro_batching_params
    yield context


@pytest.fixture(scope="module", params=[1, 8])
def handler(context, request):
    handler = ImageClassifier()

    from ts.handler_utils.micro_batching import MicroBatching

    mb_handle = MicroBatching(handler, micro_batch_size=request.param)
    handler.initialize(context)

    handler.handle = mb_handle
    handler.handle.parallelism = context.model_yaml_config["mb_parallelism"]

    yield handler

    mb_handle.shutdown()


@pytest.fixture(scope="module", params=[1, 16])
def mixed_batch(kitten_image_bytes, dog_image_bytes, request):
    batch_size = request.param
    labels = [
        "tiger_cat" if random.random() > 0.5 else "golden_retriever"
        for _ in range(batch_size)
    ]
    test_data = []
    for l in labels:
        test_data.append(
            {"data": kitten_image_bytes}
            if l == "tiger_cat"
            else {"data": dog_image_bytes}
        )
    return test_data, labels


def test_handle(context, mixed_batch, handler):
    test_data, labels = mixed_batch
    results = handler.handle(test_data, context)
    assert len(results) == len(labels)
    for l, r in zip(labels, results):
        assert l in r


def test_handle_explain(context, kitten_image_bytes, handler):
    context.explain = True
    test_data = [{"data": kitten_image_bytes, "target": 0}] * 2
    results = handler.handle(test_data, context)
    assert len(results) == 2
    assert results[0]


def test_micro_batching_handler_threads(handler):
    assert len(handler.handle.thread_groups["preprocess"]) == 1
    assert len(handler.handle.thread_groups["inference"]) == 2
    assert len(handler.handle.thread_groups["postprocess"]) == 3


def test_spin_up_down_threads(handler):
    assert len(handler.handle.thread_groups["preprocess"]) == 1
    assert len(handler.handle.thread_groups["inference"]) == 2
    assert len(handler.handle.thread_groups["postprocess"]) == 3

    new_parallelism = {
        "preprocess": 2,
        "inference": 3,
        "postprocess": 4,
    }

    handler.handle.parallelism = new_parallelism

    assert len(handler.handle.thread_groups["preprocess"]) == 2
    assert len(handler.handle.thread_groups["inference"]) == 3
    assert len(handler.handle.thread_groups["postprocess"]) == 4

    new_parallelism = {
        "preprocess": 1,
        "inference": 2,
        "postprocess": 3,
    }

    handler.handle.parallelism = new_parallelism

    assert len(handler.handle.thread_groups["preprocess"]) == 1
    assert len(handler.handle.thread_groups["inference"]) == 2
    assert len(handler.handle.thread_groups["postprocess"]) == 3
