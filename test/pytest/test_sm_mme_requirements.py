import os
import subprocess
import requests
import json
import pathlib
import pytest
import torch

import test_utils

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
MODELSTORE_DIR = os.path.join(REPO_ROOT, "model_store")
data_file_kitten = os.path.join(REPO_ROOT, "examples/image_classifier/kitten.jpg")
HF_TRANSFORMERS_EXAMPLE_DIR = os.path.join(REPO_ROOT, "examples/Huggingface_Transformers/")



def download_transformer_model():
    download_cmd = "cd " + HF_TRANSFORMERS_EXAMPLE_DIR + ";python Download_Transformer_models.py"
    subprocess.check_call(download_cmd, shell=True)

def create_transformer_mar_file():

    extra_files = os.path.join(HF_TRANSFORMERS_EXAMPLE_DIR + "Transformer_model/config.json") + "," + \
    os.path.join(HF_TRANSFORMERS_EXAMPLE_DIR, "setup_config.json") + "," + \
    os.path.join(HF_TRANSFORMERS_EXAMPLE_DIR,"Seq_classification_artifacts/index_to_name.json")

    pathlib.Path(test_utils.MODEL_STORE).mkdir(parents=True, exist_ok=True)
    
    # Generate mar file
    cmd = test_utils.model_archiver_command_builder(
        model_name="BERTSeqClassification",
        version="1.0",
        serialized_file=os.path.join(HF_TRANSFORMERS_EXAMPLE_DIR, "Transformer_model/pytorch_model.bin"),
        handler=os.path.join(HF_TRANSFORMERS_EXAMPLE_DIR, "Transformer_handler_generalized.py"),
        extra_files=extra_files,
        force=True,
    )
    cmd = cmd.split(" ")
    subprocess.run(cmd, check=True)

def test_no_model_loaded():
    """
    Validates that TorchServe returns reponse code 404 if no model is loaded.
    """

    os.makedirs(MODELSTORE_DIR, exist_ok=True)  # Create modelstore directory
    test_utils.start_torchserve(model_store=MODELSTORE_DIR)
    
    response = requests.post(url="http://localhost:8080/predictions/alexnet", data=open(data_file_kitten, 'rb'))
    assert response.status_code == 404, "Model not loaded error expected"

@pytest.mark.skipif(
    not (torch.cuda.device_count() > 0 ) and torch.cuda.is_available(),
    reason="Test to be run on GPU only",
)
def test_oom_on_model_load():
    """
    Validates that TorchServe returns reponse code 507 if there is OOM on model loading.
    """

    ## Download model
    #download_transformer_model()

    ## Create mar file
    #create_transformer_mar_file()

    # Start TorchServe
    test_utils.start_torchserve(
        no_config_snapshots=True, gen_mar=False
    )


    # Register model
    params = {
        "model_name": "BERTSeqClassification",
        "url": "BERTSeqClassification.mar",
        "batch_size": 1,
        "initial_workers": 20,
    }
    response = test_utils.register_model_with_params(params)

    assert response.status_code == 507, "OOM Error expected"

@pytest.mark.skipif(
    not (torch.cuda.device_count() > 0 ) and torch.cuda.is_available(),
    reason="Test to be run on GPU only",
)
def test_oom_on_invoke():

    ## Download model
    #download_transformer_model()
    
    ## Create mar file
    #create_transformer_mar_file()

    # Start TorchServe
    test_utils.start_torchserve(
        no_config_snapshots=True, gen_mar=False
    )


    # Register model
    params = {
        "model_name": "BERTSeqClassification",
        "url": "BERTSeqClassification.mar",
        "batch_size": 8,
        "initial_workers": 16,
    }
    response = test_utils.register_model_with_params(params)


    input_text = os.path.join(REPO_ROOT, 'examples', 'Huggingface_Transformers', 'Seq_classification_artifacts', 'sample_text_captum_input.txt')

    # Make 8 curl requests in parallel with &
    # Send multiple requests to make sure to hit OOM
    for i in range(2):
        print("Ankith !!!!!!!!!!!! i ", i)
        response = os.popen(f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && " \
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && "\
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && "\
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && "\
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && "\
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && "\
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} && "\
        f"curl http://127.0.0.1:8080/predictions/BERTSeqClassification -T {input_text} ")
        response = response.read()
    
    # If OOM is hit, we expect code 507 to be present in the response string
    lines = response.split("\n")
    output = ""
    for line in lines:
        if "code" in line:
            line = line.strip()
            output = line
            break
    assert output == '"code": 507,', "OOM Error expected"
    