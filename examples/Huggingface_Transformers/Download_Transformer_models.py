import os
import sys

import torch
import transformers
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)

print("Transformers version", transformers.__version__)
set_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transformers_model_dowloader(
    mode,
    pretrained_model_name,
    num_labels,
    do_lower_case,
    max_length,
    torchscript,
    hardware,
    batch_size,
):
    """This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
    """
    print("Download model and tokenizer", pretrained_model_name)
    # loading pre-trained model and tokenizer
    if mode == "sequence_classification":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels, torchscript=torchscript
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "question_answering":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, torchscript=torchscript
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "token_classification":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels, torchscript=torchscript
        )
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    elif mode == "text_generation":
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels, torchscript=torchscript
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )

        # NOTE : for demonstration purposes, we do not go through the fine-tune processing here.
        # A Fine_tunining process based on your needs can be added.
        # An example of  Fine_tuned model has been provided in the README.

    NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print("Creation of directory %s failed" % NEW_DIR)
    else:
        print("Successfully created directory %s " % NEW_DIR)

    print(
        "Save model and tokenizer/ Torchscript model based on the setting from setup_config",
        pretrained_model_name,
        "in directory",
        NEW_DIR,
    )
    if save_mode == "pretrained":
        model.save_pretrained(NEW_DIR)
        tokenizer.save_pretrained(NEW_DIR)
    elif save_mode == "torchscript":
        dummy_input = "This is a dummy input for torch jit trace"
        inputs = tokenizer.encode_plus(
            dummy_input,
            max_length=int(max_length),
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        model.to(device).eval()
        if hardware == "neuron":
            import torch_neuron

            input_ids = torch.cat([inputs["input_ids"]] * batch_size, 0).to(device)
            attention_mask = torch.cat([inputs["attention_mask"]] * batch_size, 0).to(
                device
            )
            traced_model = torch_neuron.trace(model, (input_ids, attention_mask))
            torch.jit.save(
                traced_model,
                os.path.join(
                    NEW_DIR,
                    "traced_{}_model_neuron_batch_{}.pt".format(model_name, batch_size),
                ),
            )
        elif hardware == "neuronx":
            import torch_neuronx

            input_ids = torch.cat([inputs["input_ids"]] * batch_size, 0).to(device)
            attention_mask = torch.cat([inputs["attention_mask"]] * batch_size, 0).to(
                device
            )
            traced_model = torch_neuronx.trace(model, (input_ids, attention_mask))
            torch.jit.save(
                traced_model,
                os.path.join(
                    NEW_DIR,
                    "traced_{}_model_neuronx_batch_{}.pt".format(
                        model_name, batch_size
                    ),
                ),
            )
        else:
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            traced_model = torch.jit.trace(model, (input_ids, attention_mask))
            torch.jit.save(traced_model, os.path.join(NEW_DIR, "traced_model.pt"))
    return


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    if len(sys.argv) > 1:
        filename = os.path.join(dirname, sys.argv[1])
    else:
        filename = os.path.join(dirname, "model-config.yaml")
    f = open(filename)
    model_yaml_config = yaml.safe_load(f)
    settings = model_yaml_config["handler"]
    mode = settings["mode"]
    model_name = settings["model_name"]
    num_labels = int(settings["num_labels"])
    do_lower_case = settings["do_lower_case"]
    max_length = settings["max_length"]
    save_mode = settings["save_mode"]
    if save_mode == "torchscript":
        torchscript = True
    else:
        torchscript = False
    hardware = settings.get("hardware")
    batch_size = int(settings.get("batch_size", "1"))

    transformers_model_dowloader(
        mode,
        model_name,
        num_labels,
        do_lower_case,
        max_length,
        torchscript,
        hardware,
        batch_size,
    )
