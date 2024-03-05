import asyncio
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from ts_scripts import marsgen as mg
from ts_scripts import tsutils as ts
from ts_scripts import utils
from ts_scripts.tsutils import generate_grpc_client_stubs

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)
MODELS_CONFIG_FILE_PATH = Path(__file__).parent.joinpath(
    "configs", "sanity_models.json"
)


async def markdown_link_checker(in_queue, out_queue, n):
    print(f"worker started {n}")
    while True:
        mdfile = await in_queue.get()
        output = []
        result = True
        cmd = f"markdown-link-check {mdfile} --config link_check_config.json"
        output.append(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
        p = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        while not p.stdout.at_eof():
            line = await p.stdout.readline()
            output.append(line.decode("utf-8"))

        status = await p.wait()
        if status != 0:
            output.append(f"## Broken links in file: {mdfile}")
            result = False
        await out_queue.put((result, output))


async def run_markdown_link_checker_on_files(files):
    results = []
    tasks = []
    in_queue = asyncio.Queue()
    out_queue = asyncio.Queue()
    for f in files:
        in_queue.put_nowait(f)

    for n in range(16):
        tasks.append(asyncio.create_task(markdown_link_checker(in_queue, out_queue, n)))

    while len(results) != len(files):
        print(len(results))
        r, output = await out_queue.get()
        results.append(r)
        for line in output:
            print(line)

    for t in tasks:
        t.cancel()

    return results


def run_markdown_link_checker():
    print("## Started markdown link checker")
    files = glob.glob("**/*.md", recursive=True)
    results = asyncio.run(run_markdown_link_checker_on_files(files))
    return all(results)


def validate_model_on_gpu():
    # A quick \ crude way of checking if model is loaded in GPU
    # Assumption is -
    # 1. GPUs on test setup are only utlizied by torchserve
    # 2. Models are successfully UNregistered between subsequent calls
    import nvgpu

    model_loaded = False
    for info in nvgpu.gpu_info():
        if info["mem_used"] > 0 and info["mem_used_percent"] > 0.0:
            model_loaded = True
            break
    return model_loaded


def load_model_to_validate():
    with open(MODELS_CONFIG_FILE_PATH) as f:
        model_list = json.load(f)
        assert isinstance(model_list, list)

    print(model_list)
    models_to_validate = {}
    for m in model_list:
        models_to_validate[m["name"]] = m

    # models_to_validate = {m["name"]: m for m in model_list}
    assert len(models_to_validate) == len(
        model_list
    ), "Model names are expected to be unique"
    return models_to_validate


def test_gpu_setup():
    is_gpu_instance = utils.is_gpu_instance()
    if is_gpu_instance:
        assert torch.cuda.is_available(), "## Ohh its NOT running on GPU !"


def run_grpc_test(model: dict):
    model_name = model["name"]
    model_inputs = model["inputs"]

    # Run gRPC sanity
    print("pass mg.mar_set=", mg.mar_set)
    mar_set_list_str = [str(s) for s in mg.mar_set]
    mar_set_str = ",".join(mar_set_list_str)
    register_model_grpc_cmd = f"python ts_scripts/torchserve_grpc_client.py register {model_name} {mar_set_str}"
    status = os.system(register_model_grpc_cmd)

    if status != 0:
        print("## Failed to register model with torchserve")
        sys.exit(1)
    else:
        print(f"## Successfully registered {model_name} model with torchserve")

    for input in model_inputs:
        infer_model_grpc_cmd = [
            "python",
            "ts_scripts/torchserve_grpc_client.py",
            "infer",
            f"{model_name}",
            f"{input}",
        ]
        p = subprocess.run(infer_model_grpc_cmd, capture_output=True, text=True)
        out = p.stdout.split("\n")
        print("\n".join(out[:50]))
        if len(out) > 50:
            print("<output clipped>")

        if p.returncode != 0:
            print(f"## Failed to run inference on {model_name} model")
            sys.exit(1)
        else:
            print(f"## Successfully ran inference on {model_name} model.")

    unregister_model_grpc_cmd = (
        f"python ts_scripts/torchserve_grpc_client.py unregister {model_name}"
    )
    status = os.system(unregister_model_grpc_cmd)

    if status != 0:
        print(f"## Failed to unregister {model_name}")
        sys.exit(1)
    else:
        print(f"## Successfully unregistered {model_name}")


def run_rest_test(model, register_model=True, unregister_model=True):
    model_name = model["name"]
    model_inputs = model["inputs"]
    model_handler = model["handler"]

    if register_model:
        response = ts.register_model(model_name)
        if response and response.status_code == 200:
            print(f"## Successfully registered {model_name} model with torchserve")
        else:
            print(f"## Failed to register {model_name} model with torchserve")
            sys.exit(1)

    # For each input execute inference n=4 times
    for input in model_inputs:
        for i in range(4):
            response = ts.run_inference(model_name, input)
            if response and response.status_code == 200:
                print(f"## Successfully ran inference on {model_name} model.")
            else:
                print(f"## Failed to run inference on {model_name} model")
                sys.exit(1)

    if torch.cuda.is_available():
        if validate_model_on_gpu():
            print(f"## Model {model_name} successfully loaded on GPU")
        else:
            sys.exit(
                f"## Something went wrong, model {model_name} did not load on GPU!!"
            )

    # skip unregistering resnet-18 model to test snapshot feature with restart
    if unregister_model:
        response = ts.unregister_model(model_name)
        if response and response.status_code == 200:
            print(f"## Successfully unregistered {model_name}")
        else:
            print(f"## Failed to unregister {model_name}")
            sys.exit(1)

    print(f"## {model_handler} handler is stable.")


def test_sanity():
    generate_grpc_client_stubs()

    print("## Started sanity tests")

    models_to_validate = load_model_to_validate()

    test_gpu_setup()

    ts_log_file = os.path.join("logs", "ts_console.log")

    os.makedirs("model_store", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    mg.mar_set = set(os.listdir("model_store"))
    started = ts.start_torchserve(log_file=ts_log_file, gen_mar=False)
    if not started:
        sys.exit(1)

    resnet18_model = models_to_validate["resnet-18"]

    models_to_validate = {
        k: v for k, v in models_to_validate.items() if k != "resnet-18"
    }

    for _, model in models_to_validate.items():
        run_grpc_test(model)
        run_rest_test(model)

    run_rest_test(resnet18_model, unregister_model=False)

    stopped = ts.stop_torchserve()
    if not stopped:
        sys.exit(1)

    # Restarting torchserve
    # This should restart with the generated snapshot and resnet-18 model should be automatically registered
    started = ts.start_torchserve(log_file=ts_log_file, gen_mar=False)
    if not started:
        sys.exit(1)

    run_rest_test(resnet18_model, register_model=False)

    stopped = ts.stop_torchserve()
    if not stopped:
        sys.exit(1)


def test_workflow_sanity():
    current_path = os.getcwd()
    ts_log_file = os.path.join("logs", "ts_console.log")
    os.makedirs("model_store", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    started = ts.start_torchserve(
        ncs=True,
        log_file=ts_log_file,
        model_store="model_store",
        workflow_store="model_store",
    )
    if not started:
        sys.exit(1)

    # Register workflow
    response = ts.register_workflow("densenet_wf")
    if response and response.status_code == 200:
        print(response.text)
    else:
        print(f"## Failed to register workflow")
        sys.exit(1)

    # Run prediction on workflow
    response = ts.workflow_prediction(
        "densenet", "examples/image_classifier/kitten.jpg"
    )
    if response and response.status_code == 200:
        print(response.text)
    else:
        print(f"## Failed to run inference on workflow - {response.text}")
        sys.exit(1)

    response = ts.unregister_workflow("densenet")
    if response and response.status_code == 200:
        print(response.text)
    else:
        print(f"## Failed to unregister workflow")
        sys.exit(1)

    stopped = ts.stop_torchserve()
    if not stopped:
        sys.exit(1)


def test_markdown_files():
    links_ok = run_markdown_link_checker()
    if not links_ok:
        print("##WARNING : Broken links in docs.")
    return links_ok
