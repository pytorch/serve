import argparse
import io
import logging
from pathlib import Path

import torch

from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

CURR_FILE_PATH = Path(__file__).parent.absolute()
REPO_ROOT_DIR = CURR_FILE_PATH.parents[3]
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "image_classifier", "resnet_18")
TEST_DATA = REPO_ROOT_DIR.joinpath("examples", "image_classifier", "kitten.jpg")
MODEL_PT_FILE = "resnet-18.pt"


def prepare_data(batch_size):
    """
    Function to prepare data based on the desired batch size
    """
    f = io.open(TEST_DATA, "rb", buffering=0)
    read_data = f.read()
    data = []
    for i in range(batch_size):
        tmp = {}
        tmp["data"] = read_data
        data.append(tmp)
    return data


def test_resnet18(batch_size=1):
    # Define your handler
    handler = ImageClassifier()

    # Context definition
    ctx = MockContext(
        model_pt_file=MODEL_PT_FILE,
        model_dir=EXAMPLE_ROOT_DIR.as_posix(),
        model_file=None,
    )

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)
    handler.context = ctx

    data = prepare_data(batch_size)

    # Here we are using the BaseHandler's handle method. You can define your own
    result = handler.handle(data, ctx)
    logger.info(f"Result is {result}")

    # Can be used with pytest
    value = max(result[0], key=result[0].get)
    assert value == "tabby"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size for testing inference",
    )
    args = parser.parse_args()
    test_resnet18(args.batch_size)
