import json
import shutil
from pathlib import Path
from queue import Empty
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import pytest
import requests
import test_utils
import torch
from model_archiver import ModelArchiverConfig
from test_data.streaming.stream_handler import StreamingHandler

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = Path(__file__).parent


@pytest.fixture(scope="module")
def model_name():
    yield "streaming_handler"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name):
    mar_file_path = Path(work_dir).joinpath(model_name + ".mar")

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        model_file=CURR_FILE_PATH.joinpath(
            "test_data", "streaming", "fake_streaming_model.py"
        ).as_posix(),
        handler=CURR_FILE_PATH.joinpath(
            "test_data", "streaming", "stream_handler.py"
        ).as_posix(),
        serialized_file=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        archive_format="default",
        config_file=CURR_FILE_PATH.joinpath(
            "test_data", "streaming", "model_config.yaml"
        ).as_posix(),
        extra_files=None,
    )

    with patch("archiver.ArgParser.export_model_args_parser", return_value=config):
        # Using ZIP_STORED instead of ZIP_DEFLATED reduces test runtime from 54 secs to 10 secs
        with patch(
            "model_archiver.model_packaging_utils.zipfile.ZipFile",
            lambda x, y, _: ZipFile(x, y, ZIP_STORED),
        ):
            model_archiver.generate_model_archive()

            assert mar_file_path.exists()

            yield mar_file_path.as_posix()

    # Clean up files
    # mar_file_path.unlink(missing_ok=True)


@pytest.fixture(scope="module", name="model_name_and_stdout")
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
        ("batch_size", "2"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name, torchserve

    test_utils.unregister_model(model_name)


@pytest.mark.skip(reason="Skipping this test for now")
def test_echo_stream_inference(model_name_and_stdout):
    model_name, _ = model_name_and_stdout
    responses = []
    data = [
        {
            "prompt": "The capital of France",
            "max_new_tokens": 5,
        },
        {
            "prompt": "Europe is",
            "max_new_tokens": 10,
        },
        {
            "prompt": "The US are",
            "max_new_tokens": 15,
        },
        {
            "prompt": "When travelling to NYC",
            "max_new_tokens": 5,
        },
    ]
    for d in data:
        res = requests.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            data=json.dumps(d),
            stream=True,
        )

        responses.append(res)
    assert all(r.headers["Transfer-Encoding"] == "chunked" for r in responses)

    all_predictions = []
    for idx, d in enumerate(data):
        prediction = []
        for chunk in responses[idx].iter_content(chunk_size=None):
            if chunk:
                prediction.append(chunk.decode("utf-8"))

        all_predictions.append("".join(json.loads(p)["text"] for p in prediction))

    assert all_predictions[0] == "The capital of France, Paris, is home"
    assert (
        all_predictions[1] == "Europe is a country of immigrants, and it is a country"
    )
    assert (
        all_predictions[2]
        == "The US are not going to be able to do that. They're going to have to"
    )
    assert all_predictions[3] == "When travelling to NYC, I was able to"


@pytest.mark.skip(reason="Skipping this test for now")
def test_decoding_stage(monkeypatch):
    monkeypatch.syspath_prepend((CURR_FILE_PATH / "test_data" / "streaming"))

    handler = StreamingHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=(CURR_FILE_PATH / "test_data" / "streaming").as_posix(),
        model_file="fake_streaming_model.py",
    )
    ctx.model_yaml_config["handler"] = {"modelId": "gpt2"}

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)

    handler.context = ctx

    device = next(iter(handler.model.parameters())).device

    ctx.cache = {
        "id1": {
            "encoded": {
                "input_ids": torch.randint(42, (1, 5), device=device),
                "attention_mask": torch.ones((1, 5), dtype=int, device=device),
                "past_key_values": None,
            },
        },
        "id2": {
            "encoded": {
                "input_ids": torch.randint(42, (1, 8), device=device),
                "attention_mask": torch.ones((1, 8), dtype=int, device=device),
                "past_key_values": None,
            }
        },
    }
    ctx.cache["id1"]["encoded"]["attention_mask"][0, :2] = 0

    res = handler._run_prefill("id1")
    res = handler._run_prefill("id2")

    res = handler._run_decode(["id1"])

    assert len(res["id1"]["ids"]) == 1
    assert type(res["id1"]["text"]) == str

    assert ctx.cache["id1"]["encoded"]["input_ids"].size()[-1] == 5
    assert ctx.cache["id1"]["encoded"]["attention_mask"].size()[-1] == 5

    res = handler._run_decode(["id1", "id2"])
    assert ctx.cache["id1"]["encoded"]["input_ids"].size()[-1] == 10
    assert ctx.cache["id1"]["encoded"]["attention_mask"].size()[-1] == 10

    assert ctx.cache["id2"]["encoded"]["input_ids"].size()[-1] == 10
    assert ctx.cache["id2"]["encoded"]["attention_mask"].size()[-1] == 10

    res = handler._run_decode(["id1"])
    assert ctx.cache["id1"]["encoded"]["input_ids"].size()[-1] == 7
    assert ctx.cache["id1"]["encoded"]["attention_mask"].size()[-1] == 7

    res = handler._run_decode(["id1", "id2"])
    assert ctx.cache["id1"]["encoded"]["input_ids"].size()[-1] == 11
    assert ctx.cache["id1"]["encoded"]["attention_mask"].size()[-1] == 11

    assert ctx.cache["id2"]["encoded"]["input_ids"].size()[-1] == 11
    assert ctx.cache["id2"]["encoded"]["attention_mask"].size()[-1] == 11


@pytest.mark.skip(reason="Skipping this test for now")
def test_closed_connection(model_name_and_stdout):
    model_name, stdout = model_name_and_stdout

    # Empty queue
    while not stdout.empty():
        stdout.get_nowait()

    data = {
        "prompt": "The capital of France",
        "max_new_tokens": 500,
    }

    with requests.Session() as s:
        res = s.post(
            url=f"http://localhost:8080/predictions/{model_name}",
            data=json.dumps(data),
            stream=True,
        )

        for chunk in res.iter_content(chunk_size=None):
            # Close connection after the first id has been received
            break

    lines = []
    while True:
        try:
            lines.append(stdout.get(timeout=5))
        except Empty:
            assert 0, "Queue timed out"

        if "Connection to client got closed; Removing job:" in lines[-1]:
            break

    # We expect the model to only run two times at most due to the closed connection
    assert len(list(filter(lambda x: "Backend received inference at" in x, lines))) <= 2
