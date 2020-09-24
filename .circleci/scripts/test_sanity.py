import os
import sys
import nvgpu
import glob

from scripts import install_utils

models_to_validate = {
    "fastrcnn": {
        "inputs": ["examples/object_detector/persons.jpg"],
        "handler": "object_detector"
    },
    "fcn_resnet_101":{
        "inputs": ["docs/images/blank_image.jpg", "examples/image_segmenter/fcn/persons.jpg"],
        "handler": "image_segmenter"
    },
    "my_text_classifier_v2": {
        "inputs": ["examples/text_classification/sample_text.txt"],
        "handler": "text_classification"
    },
    "resnet-18": {
        "inputs": ["examples/image_classifier/kitten.jpg"],
        "handler": "image_classifier"
    },
    "my_text_classifier_scripted_v2": {
        "inputs": ["examples/text_classification/sample_text.txt"],
        "handler": "text_classification"
    },
    "alexnet_scripted": {
        "inputs": ["examples/image_classifier/kitten.jpg"],
        "handler": "image_classifier"
    },
    "fcn_resnet_101_scripted": {
        "inputs": ["examples/image_segmenter/fcn/persons.jpg"],
        "handler": "image_segmenter"
    },
    "roberta_qa_no_torchscript": {
        "inputs": ["examples/Huggingface_Transformers/QA_artifacts/sample_text.txt"],
        "handler": "custom"
    },
    "bert_token_classification_no_torchscript":{
        "inputs": ["examples/Huggingface_Transformers/Token_classification_artifacts/sample_text.txt"],
        "handler":"custom"
    },
    "bert_seqc_without_torchscript":{
        "inputs": ["examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt"],
        "handler": "custom"
    }
}
ts_log_file = os.path.join("logs", "ts_console.log")
is_gpu_instance = install_utils.is_gpu_instance()


def run_markdown_link_checker():
    success = True
    for mdfile in glob.glob("**/*.md", recursive=True):
        status = os.system(f"markdown-link-check {mdfile} --config link_check_config.json")
        if status != 0:
            print(f'Broken links in {mdfile}')
            success = False
    if not success:
        sys.exit("Markdown Link Checker Failed")


def validate_model_on_gpu():
    # A quick \ crude way of checking if model is loaded in GPU
    # Assumption is -
    # 1. GPU on test setup is only utlizied by torchserve
    # 2. Models are successfully UNregistered between subsequent calls
    model_loaded = False
    for info in nvgpu.gpu_info():
        if info['mem_used'] > 0 and info['mem_used_percent'] > 0.0:
            model_loaded = True
            break
    return model_loaded


os.mkdir('model_store')
os.mkdir('logs')

if is_gpu_instance:
    import torch
    if not torch.cuda.is_available():
      print("Ohh its NOT running on GPU!!")
      sys.exit(1)

install_utils.start_torchserve(log_file=ts_log_file)

for model, model_config in models_to_validate.items():
    install_utils.register_model(model)
    inputs = model_config['inputs']

    for input in inputs:
        install_utils.run_inference(model, input)

    if is_gpu_instance:
        if validate_model_on_gpu():
            print(f"Model {model} successfully loaded on GPU")
        else:
            print(f"Something went wrong, model {model} did not load on GPU!!")
            sys.exit(1)

    #skip unregistering resnet-18 model to test snapshot feature with restart
    if model != "resnet-18":
        install_utils.unregister_model(model)

    print(f"{model_config['handler']} default handler is stable.")

install_utils.stop_torchserve()

# Restarting torchserve
# This should restart with the generated snapshot and resnet-18 model should be automatically registered
install_utils.start_torchserve(log_file=ts_log_file)

install_utils.run_inference("resnet-18", models_to_validate["resnet-18"]["inputs"][0])

install_utils.stop_torchserve()

run_markdown_link_checker()