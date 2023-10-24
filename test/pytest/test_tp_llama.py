import os
import sys
from collections import OrderedDict
from multiprocessing import Process, Queue
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent
LLAMA_PATH = CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "tp_llama"
sys.path.append(LLAMA_PATH.as_posix())

converted_checkpoints_path = "llama/converted_checkpoints"

YAML_CONFIG = f"""
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 200
responseTimeout: 300
parallelType: "tp"
deviceType: "gpu"

torchrun:
    nproc-per-node: 1

handler:
    converted_ckpt_dir: "{converted_checkpoints_path}"
    tokenizer_path: "llama/tokenizer.model"
    model_args_path: "llama/model_args.json"
    max_new_tokens: 50
    temperature: 0.0
    top_p: 0.9
    manual_seed: 40
    mode: "text_completion" #choices are text_completion, chat
"""


def call_handler(rank: int, world_size: int, queue: Queue, yaml_path: str):
    from llama_handler import LlamaHandler

    handler = LlamaHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=LLAMA_PATH.as_posix(),
    )

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    ctx.model_yaml_config = config

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)

    handler.context = ctx

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
            {"data": {"prompt": "The capital of France"}},
            {"data": {"prompt": "what is the recipes for Mayonnaise?"}},
        ]
    )
    queue.put([handler.tokenizer.decode(s) for s in sequences])

    # Send the shorter sequence again to to see if it generates the same output now without padding
    handler.context.request_ids = OrderedDict(((0, "id3"),))

    sequences = run_inference(
        [
            {"data": {"prompt": "The capital of France"}},
        ]
    )
    queue.put([handler.tokenizer.decode(s) for s in sequences])


@pytest.mark.skipif(
    not (LLAMA_PATH / converted_checkpoints_path).exists(),
    reason=f"Required files are not present {(LLAMA_PATH / converted_checkpoints_path).as_posix()}",
)
def test_tensor_parallel_llama(tmp_path):
    world_size = 2

    model_config_yaml = tmp_path / "model-config.yaml"
    model_config_yaml.write_text(YAML_CONFIG)

    q = Queue()

    procs = [
        Process(
            target=call_handler,
            args=(
                rank,
                world_size,
                q,
                model_config_yaml,
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

    assert len(results) == 4

    assert (
        results[0][0]
        == ", Paris, is a city of romance, art, and culture. It is also a city of fashion, food, and fun. Paris is a city that has something for everyone.\nParis is a city that is full of history."
    )
    assert (
        results[0][1]
        == "\nI have a recipe for mayonnaise that I use all the time. It is very easy and tastes great.\n1. In a bowl, whisk together the egg yolks, mustard, lemon ju"
    )

    assert (
        results[2][0]
        == ", Paris, is a city of romance, art, and culture. It is also a city of fashion, food, and fun. Paris is a city that has something for everyone.\nParis is a city that is full of history."
    )


@patch("llama_handler.torch.distributed.init_process_group")
@patch("llama_handler.Llama.build")
def test_handler(build, ipg, tmp_path, mocker):
    BSZ = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    build.return_value.model.parameters.return_value = torch.nn.Linear(
        1, 1, device=device
    ).parameters()
    build.return_value.model.layers = [mocker.MagicMock(name="layer")]
    build.return_value.model.layers[0].attention.cache_k.size.return_value = BSZ

    from llama_handler import LlamaHandler

    handler = LlamaHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=LLAMA_PATH.as_posix(),
    )
    model_config_yaml = tmp_path / "model-config.yaml"
    model_config_yaml.write_text(YAML_CONFIG)

    with open(model_config_yaml, "r") as f:
        config = yaml.safe_load(f)

    ctx.model_yaml_config = config

    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)

    handler.context = ctx

    ctx.cache = {
        "id1": {
            "encoded": torch.randint(42, (3,), device=device),
        },
        "id2": {
            "encoded": torch.randint(42, (8,), device=device),
        },
    }

    handler.tokenizer = mocker.MagicMock(name="tokenizer")
    handler.tokenizer.eos_id = 2

    def forward(input_ids, *args, **kwargs):
        return torch.rand((input_ids.size(0), input_ids.size(1), 32), device=device)

    build.return_value.model.forward.side_effect = forward

    res1 = handler._run_prefill("id1")
    res2 = handler._run_prefill("id2")

    assert len(res1["id1"]["ids"]) == 1
    assert len(res2["id2"]["ids"]) == 1

    assert ctx.cache["id1"]["encoded"].size(-1) == 4
    assert ctx.cache["id2"]["encoded"].size(-1) == 9

    res = handler._run_decode(["id1"])

    assert len(res["id1"]["ids"]) == 1
    assert ctx.cache["id1"]["encoded"].size(-1) == 5

    res = handler._run_decode(["id1", "id2"])
    assert ctx.cache["id1"]["encoded"].size(-1) == 6
    assert ctx.cache["id2"]["encoded"].size(-1) == 10

    res = handler._run_decode(["id1"])
    assert ctx.cache["id1"]["encoded"].size(-1) == 7

    res = handler._run_decode(["id2"])
    assert ctx.cache["id2"]["encoded"].size(-1) == 11

    res = handler._run_decode(["id1", "id2"])
    assert ctx.cache["id1"]["encoded"].size(-1) == 8
    assert ctx.cache["id2"]["encoded"].size(-1) == 12


def test_clean_cache(mocker):
    from llama_handler import LlamaHandler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    handler = LlamaHandler()

    handler.context = mocker.MagicMock(name="context")
    handler.context.request_ids = {k: f"id{k+1}" for k in range(3)}

    handler.batch_idx_to_req_ids = ["id5", "id1", None, "id2", "id3", None, "id4", None]
    handler.max_bsz = len(handler.batch_idx_to_req_ids)

    handler.model = mocker.MagicMock(name="model")
    handler.model.layers = [mocker.MagicMock(name="layer")]
    handler.model.layers[0].attention.cache_k = -1 * torch.ones(
        (handler.max_bsz, 1), dtype=torch.long, device=device
    )
    handler.model.layers[0].attention.cache_k[0] = 5
    handler.model.layers[0].attention.cache_k[1] = 1
    handler.model.layers[0].attention.cache_k[3] = 2
    handler.model.layers[0].attention.cache_k[4] = 3
    handler.model.layers[0].attention.cache_k[6] = 4

    handler.model.layers[0].attention.cache_v = handler.model.layers[
        0
    ].attention.cache_k.clone()

    handler._clean_cache()

    assert handler.batch_idx_to_req_ids.index(None) == 3
    assert handler.batch_idx_to_req_ids[0] == "id3"
    assert handler.batch_idx_to_req_ids[1] == "id1"
    assert handler.batch_idx_to_req_ids[2] == "id2"

    assert handler.model.layers[0].attention.cache_k[0] == 3
    assert handler.model.layers[0].attention.cache_k[1] == 1
    assert handler.model.layers[0].attention.cache_k[2] == 2

    assert handler.model.layers[0].attention.cache_k.equal(
        handler.model.layers[0].attention.cache_v
    )


def test_vacate_kv_cache_before_prefill(mocker):
    from llama_handler import LlamaHandler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    handler = LlamaHandler()

    handler.batch_idx_to_req_ids = ["id5", "id1", "id2", "id3", None, None]
    handler.max_bsz = len(handler.batch_idx_to_req_ids)

    handler.model = mocker.MagicMock(name="model")
    handler.model.layers = [mocker.MagicMock(name="layer")]
    handler.model.layers[0].attention.cache_k = -1 * torch.ones(
        (handler.max_bsz, 1), dtype=torch.long, device=device
    )
    handler.model.layers[0].attention.cache_k[0] = 5
    handler.model.layers[0].attention.cache_k[1] = 1
    handler.model.layers[0].attention.cache_k[2] = 2
    handler.model.layers[0].attention.cache_k[3] = 3

    handler.model.layers[0].attention.cache_v = handler.model.layers[
        0
    ].attention.cache_k.clone()

    handler._vacate_kv_cache_before_prefill()

    assert handler.batch_idx_to_req_ids.index(None) == 0
    assert handler.batch_idx_to_req_ids[0] == None
    assert handler.batch_idx_to_req_ids[1] == "id1"
    assert handler.batch_idx_to_req_ids[2] == "id2"
    assert handler.batch_idx_to_req_ids[3] == "id3"
    assert handler.batch_idx_to_req_ids[4] == "id5"

    assert handler.model.layers[0].attention.cache_k[1] == 1
    assert handler.model.layers[0].attention.cache_k[2] == 2
    assert handler.model.layers[0].attention.cache_k[3] == 3
    assert handler.model.layers[0].attention.cache_k[4] == 5

    assert handler.model.layers[0].attention.cache_k.equal(
        handler.model.layers[0].attention.cache_v
    )


def test_rearrange_kv_cache_for_decode(mocker):
    from llama_handler import LlamaHandler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    handler = LlamaHandler()

    handler.batch_idx_to_req_ids = ["id5", "id1", None, "id2", "id3", None, "id4", None]
    handler.max_bsz = len(handler.batch_idx_to_req_ids)

    handler.model = mocker.MagicMock(name="model")
    handler.model.layers = [mocker.MagicMock(name="layer")]
    handler.model.layers[0].attention.cache_k = -1 * torch.ones(
        (handler.max_bsz, 1), dtype=torch.long, device=device
    )
    handler.model.layers[0].attention.cache_k[0] = 5
    handler.model.layers[0].attention.cache_k[1] = 1
    handler.model.layers[0].attention.cache_k[3] = 2
    handler.model.layers[0].attention.cache_k[4] = 3
    handler.model.layers[0].attention.cache_k[6] = 4

    handler.model.layers[0].attention.cache_v = handler.model.layers[
        0
    ].attention.cache_k.clone()

    handler._rearrange_kv_cache_for_decode(["id4", "id1", "id3"])

    assert handler.batch_idx_to_req_ids[0] == "id4"
    assert handler.batch_idx_to_req_ids[1] == "id1"
    assert handler.batch_idx_to_req_ids[2] == "id3"

    assert set(handler.batch_idx_to_req_ids[3:5]) == set(["id2", "id5"])
    assert handler.batch_idx_to_req_ids[5:] == 3 * [
        None,
    ]

    assert handler.model.layers[0].attention.cache_k[0] == 4
    assert handler.model.layers[0].attention.cache_k[1] == 1
    assert handler.model.layers[0].attention.cache_k[2] == 3
    assert handler.model.layers[0].attention.cache_k[3] in [2, 5]
    assert handler.model.layers[0].attention.cache_k[4] in [2, 5]

    assert handler.model.layers[0].attention.cache_k.equal(
        handler.model.layers[0].attention.cache_v
    )


def test_prepare_model_inputs(mocker):
    SEQLEN = 32
    HEAD_DIM = 16
    HEAD_NUM = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from llama_handler import LlamaHandler

    handler = LlamaHandler()
    handler.device = device

    handler.batch_idx_to_req_ids = ["id5", "id1", None, "id2", "id3", None, "id4", None]
    handler.max_bsz = len(handler.batch_idx_to_req_ids)

    handler.model = mocker.MagicMock(name="model")
    handler.model.layers = [mocker.MagicMock(name="layer")]
    handler.model.layers[0].attention.cache_k = -1 * torch.ones(
        (handler.max_bsz, SEQLEN, HEAD_NUM, HEAD_DIM), dtype=torch.long, device=device
    )
    handler.model.layers[0].attention.cache_k[0, 5:, ...] = 5
    handler.model.layers[0].attention.cache_k[1, 5:, ...] = 1
    handler.model.layers[0].attention.cache_k[3, 5 : 5 + 2, ...] = 2
    handler.model.layers[0].attention.cache_k[4, 5 : 5 + 3, ...] = 3
    handler.model.layers[0].attention.cache_k[6, 5 : 5 + 4, ...] = 4

    handler.model.layers[0].attention.cache_v = handler.model.layers[
        0
    ].attention.cache_k.clone()

    handler.context = mocker.MagicMock(name="context")
    handler.context.cache = {f"id{idx}": {} for idx in [2, 3, 4]}
    handler.context.cache["id2"]["encoded"] = 200 * torch.ones((1, 2), device=device)
    handler.context.cache["id2"]["padding"] = 5
    handler.context.cache["id4"]["encoded"] = 400 * torch.ones((1, 4), device=device)
    handler.context.cache["id4"]["padding"] = 5
    handler.context.cache["id3"]["encoded"] = 300 * torch.ones((1, 3), device=device)
    handler.context.cache["id3"]["padding"] = 5

    handler.tokenizer = mocker.MagicMock(name="tokenizer")
    handler.tokenizer.eos_id = 2

    tokens, padding = handler._prepare_model_inputs(["id2", "id4", "id3"])

    assert padding.equal(torch.tensor([2, 0, 1], dtype=torch.long, device=device))
    assert all(tokens[0, :2] == 2)
    assert all(tokens[0, 2:] == 200)
    assert all(tokens[1, :] == 400)
    assert all(tokens[2, :1] == 2)
    assert all(tokens[2, 1:] == 300)

    assert handler.batch_idx_to_req_ids[0] == "id2"
    assert handler.batch_idx_to_req_ids[1] == "id4"
    assert handler.batch_idx_to_req_ids[2] == "id3"

    assert all(handler.model.layers[0].attention.cache_k[0, 2:2:+2, ...] == 2)
    assert all(handler.model.layers[0].attention.cache_k[0, 0:0:+4, ...] == 4)
    assert all(handler.model.layers[0].attention.cache_k[0, 1:1:+3, ...] == 3)

    assert handler.model.layers[0].attention.cache_k.equal(
        handler.model.layers[0].attention.cache_v
    )
