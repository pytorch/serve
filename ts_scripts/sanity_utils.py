import os
import sys
import nvgpu
import glob


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts import tsutils as ts
from ts_scripts.tsutils import generate_grpc_client_stubs
from ts_scripts import utils


def run_markdown_link_checker():
    print("## Started markdown link checker")
    result = True
    for mdfile in glob.glob("**/*.md", recursive=True):
        cmd = f"markdown-link-check {mdfile} --config link_check_config.json"
        print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
        status = os.system(cmd)
        if status != 0:
            print(f"## Broken links in file: {mdfile}")
            result = False
    return result


def validate_model_on_gpu():
    # A quick \ crude way of checking if model is loaded in GPU
    # Assumption is -
    # 1. GPUs on test setup are only utlizied by torchserve
    # 2. Models are successfully UNregistered between subsequent calls
    model_loaded = False
    for info in nvgpu.gpu_info():
        if info["mem_used"] > 0 and info["mem_used_percent"] > 0.0:
            model_loaded = True
            break
    return model_loaded


def test_sanity():
    generate_grpc_client_stubs()

    import pathlib
    pathlib.Path(__file__).parent.absolute()

    print("## Started sanity tests")

    resnet18_model = {"name": "resnet-18", "inputs": ["examples/image_classifier/kitten.jpg"],
                      "handler": "image_classifier"}
    models_to_validate = [
        {"name": "fastrcnn", "inputs": ["examples/object_detector/persons.jpg"], "handler": "object_detector"},
        {"name": "fcn_resnet_101",
         "inputs": ["docs/images/blank_image.jpg", "examples/image_segmenter/persons.jpg"],
         "handler": "image_segmenter"},
        {"name": "my_text_classifier_v2", "inputs": ["examples/text_classification/sample_text.txt"],
         "handler": "text_classification"},
        resnet18_model,
        {"name": "my_text_classifier_scripted_v2", "inputs": ["examples/text_classification/sample_text.txt"],
         "handler": "text_classification"},
        {"name": "alexnet_scripted", "inputs": ["examples/image_classifier/kitten.jpg"], "handler": "image_classifier"},
        {"name": "fcn_resnet_101_scripted", "inputs": ["examples/image_segmenter/persons.jpg"],
         "handler": "image_segmenter"},
        {"name": "distillbert_qa_no_torchscript",
         "inputs": ["examples/Huggingface_Transformers/QA_artifacts/sample_text.txt"], "handler": "custom"},
        {"name": "bert_token_classification_no_torchscript",
         "inputs": ["examples/Huggingface_Transformers/Token_classification_artifacts/sample_text.txt"],
         "handler": "custom"},
        {"name": "bert_seqc_without_torchscript",
         "inputs": ["examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt"],
         "handler": "custom"}
    ]
    ts_log_file = os.path.join("logs", "ts_console.log")
    is_gpu_instance = utils.is_gpu_instance()

    os.makedirs("model_store", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if is_gpu_instance:
        import torch
        if not torch.cuda.is_available():
            sys.exit("## Ohh its NOT running on GPU !")

    started = ts.start_torchserve(log_file=ts_log_file)
    if not started:
        sys.exit(1)

    for model in models_to_validate:
        model_name = model["name"]
        model_inputs = model["inputs"]
        model_handler = model["handler"]

        # Run gRPC sanity
        register_model_grpc_cmd = f"python ts_scripts/torchserve_grpc_client.py register {model_name}"
        status = os.system(register_model_grpc_cmd)

        if status != 0:
            print("## Failed to register model with torchserve")
            sys.exit(1)
        else:
            print(f"## Successfully registered {model_name} model with torchserve")

        for input in model_inputs:
            infer_model_grpc_cmd = f"python ts_scripts/torchserve_grpc_client.py infer {model_name} {input}"
            status = os.system(infer_model_grpc_cmd)
            if status != 0:
                print(f"## Failed to run inference on {model_name} model")
                sys.exit(1)
            else:
                print(f"## Successfully ran inference on {model_name} model.")

        unregister_model_grpc_cmd = f"python ts_scripts/torchserve_grpc_client.py unregister {model_name}"
        status = os.system(unregister_model_grpc_cmd)

        if status != 0:
            print(f"## Failed to unregister {model_name}")
            sys.exit(1)
        else:
            print(f"## Successfully unregistered {model_name}")

        # Run REST sanity
        response = ts.register_model(model_name)
        if response and response.status_code == 200:
            print(f"## Successfully registered {model_name} model with torchserve")
        else:
            print("## Failed to register model with torchserve")
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

        if is_gpu_instance:
            if validate_model_on_gpu():
                print(f"## Model {model_name} successfully loaded on GPU")
            else:
                sys.exit(f"## Something went wrong, model {model_name} did not load on GPU!!")

        # skip unregistering resnet-18 model to test snapshot feature with restart
        if model != resnet18_model:
            response = ts.unregister_model(model_name)
            if response and response.status_code == 200:
                print(f"## Successfully unregistered {model_name}")
            else:
                print(f"## Failed to unregister {model_name}")
                sys.exit(1)

        print(f"## {model_handler} handler is stable.")

    stopped = ts.stop_torchserve()
    if not stopped:
        sys.exit(1)

    # Restarting torchserve
    # This should restart with the generated snapshot and resnet-18 model should be automatically registered
    started = ts.start_torchserve(log_file=ts_log_file)
    if not started:
        sys.exit(1)

    response = ts.run_inference(resnet18_model["name"], resnet18_model["inputs"][0])
    if response and response.status_code == 200:
        print(f"## Successfully ran inference on {resnet18_model['name']} model.")
    else:
        print(f"## Failed to run inference on {resnet18_model['name']} model")
        sys.exit(1)

    stopped = ts.stop_torchserve()
    if not stopped:
        sys.exit(1)

    links_ok = run_markdown_link_checker()
    if not links_ok:
       print("##WARNING : Broken links in docs.")
