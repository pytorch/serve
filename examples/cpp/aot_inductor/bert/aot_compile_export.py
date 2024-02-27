import os
import sys

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

set_seed(1)
# PT2.2 has limitation on the max
MAX_BATCH_SIZE = 15
MAX_SEQ_LENGTH = 511


def transformers_model_dowloader(
    mode,
    pretrained_model_name,
    num_labels,
    do_lower_case,
    max_length,
    batch_size,
):
    print("Download model and tokenizer", pretrained_model_name)
    # loading pre-trained model and tokenizer
    if mode == "sequence_classification":
        config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            torchscript=False,
            return_dict=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, do_lower_case=do_lower_case
        )
    else:
        sys.exit(f"mode={mode} has not been implemented in this cpp example yet.")

    NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print("Creation of directory %s failed" % NEW_DIR)
    else:
        print("Successfully created directory %s " % NEW_DIR)

    print(
        "Save model and tokenizer model based on the setting from setup_config",
        pretrained_model_name,
        "in directory",
        NEW_DIR,
    )

    model.save_pretrained(NEW_DIR)
    tokenizer.save_pretrained(NEW_DIR)

    with torch.no_grad():
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device=device)
        dummy_input = "This is a dummy input for torch compile export"
        inputs = tokenizer.encode_plus(
            dummy_input,
            max_length=max_length,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = torch.cat([inputs["input_ids"]] * batch_size, 0).to(device)
        attention_mask = torch.cat([inputs["attention_mask"]] * batch_size, 0).to(
            device
        )
        batch_dim = torch.export.Dim("batch", min=1, max=MAX_BATCH_SIZE)
        seq_len_dim = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LENGTH)
        torch._C._GLIBCXX_USE_CXX11_ABI = True
        model_so_path = torch._export.aot_compile(
            model,
            (input_ids, attention_mask),
            dynamic_shapes={
                "input_ids": (batch_dim, seq_len_dim),
                "attention_mask": (batch_dim, seq_len_dim),
            },
            options={
                "aot_inductor.output_path": os.path.join(os.getcwd(), "bert-seq.so"),
                "max_autotune": True,
            },
        )

    return


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    if len(sys.argv) > 1:
        filename = os.path.join(dirname, sys.argv[1])
    else:
        filename = os.path.join(dirname, "model-config.yaml")
    with open(filename, "r") as f:
        settings = yaml.safe_load(f)

    mode = settings["handler"]["mode"]
    model_name = settings["handler"]["model_name"]
    num_labels = int(settings["handler"]["num_labels"])
    do_lower_case = bool(settings["handler"]["do_lower_case"])
    max_length = int(settings["handler"]["max_length"])
    batch_size = int(settings["batchSize"])
    transformers_model_dowloader(
        mode,
        model_name,
        num_labels,
        do_lower_case,
        max_length,
        batch_size,
    )
