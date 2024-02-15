import json
import os
import sys
from collections import OrderedDict
from multiprocessing import Process, Queue
from pathlib import Path

import pytest
import torch
import yaml

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
PIPPY_EXAMPLE_PATH = (
    CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "Huggingface_pippy"
)


@pytest.fixture
def add_paths():
    sys.path.append(PIPPY_EXAMPLE_PATH.as_posix())
    yield
    sys.path.pop()


def call_handler(rank: int, world_size: int, queue: Queue):
    from pippy_handler import TransformersSeqClassifierHandler

    handler = TransformersSeqClassifierHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=PIPPY_EXAMPLE_PATH.as_posix(),
    )
    model_config_yaml = PIPPY_EXAMPLE_PATH / "model-config.yaml"

    with open(model_config_yaml, "r") as f:
        config = yaml.safe_load(f)

    ctx.model_yaml_config = config
    ctx.request_ids = {0: "0"}

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"

    torch.manual_seed(42)

    handler.initialize(ctx)

    handler.context = ctx

    assert ("cuda:0" if torch.cuda.is_available() else "cpu") == str(handler.device)

    def run_inference(requests):
        results = []
        for _ in range(50):
            out = handler.preprocess(requests)
            out = handler.inference(out)
            ret = handler.postprocess(out)

            results.append([r["ids"][0] for r in ret])

        return [[r[i] for r in results] for i in range(len(requests))]

    # Combine two sequences
    handler.context.request_ids = OrderedDict(((0, "id1"), (1, "id2")))

    sequences = run_inference(
        [
            {"data": json.dumps({"prompt": "The capital of France"})},
            {"data": json.dumps({"prompt": "The capital of Germany"})},
        ]
    )
    print(f"{sequences=}")
    queue.put([handler.tokenizer.decode(s) for s in sequences])


def test_handler(add_paths):
    world_size = 2
    q = Queue()

    procs = [
        Process(
            target=call_handler,
            args=(
                rank,
                world_size,
                q,
            ),
        )
        for rank in range(world_size)
    ]

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    results = []
    while not q.empty():
        results.append(q.get())
