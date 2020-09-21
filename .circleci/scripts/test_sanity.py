import os

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

os.mkdir('model_store')

install_utils.install_bert_dependencies()

install_utils.start_torchserve()

for model, model_config in models_to_validate.items():
    install_utils.register_model(model)
    inputs = model_config['inputs']

    for input in inputs:
        install_utils.run_inference(model, input)

    # ToDo: gpu validate_model_on_gpu.py

    #skip unregistering resnet-18 model to test snapshot feature with restart
    if model != "resnet-18":
        install_utils.unregister_model(model)

    print(f"{model_config['handler']} default handler is stable.")

install_utils.stop_torchserve()

# Restarting torchserve
# This should restart with the generated snapshot and resnet-18 model should be automatically registered
install_utils.start_torchserve()

install_utils.run_inference("resnet-18", models_to_validate["resnet-18"]["inputs"][0])

install_utils.stop_torchserve()

install_utils.run_markdown_link_checker()