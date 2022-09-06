"""
This script creates a DLRM model and packs it into a TorchServe mar file
"""

import os

import torch
from dlrm_factory import DLRMFactory

MODEL_PT_FILE = "dlrm.pt"


def create_pt_file(output_file: str) -> None:
    module = DLRMFactory()

    torch.save(module.cpu().state_dict(), output_file)


def main():
    print(f"Creating DLRM model and saving state_dict to {MODEL_PT_FILE}")
    create_pt_file(MODEL_PT_FILE)

    cmd = [
        "torch-model-archiver",
        "--model-name dlrm",
        "--version 1.0",
        f"--serialized-file {MODEL_PT_FILE}",
        "--model-file dlrm_factory.py",
        "--extra-files dlrm_model_config.py",
        "--handler dlrm_handler.py",
        "--force",
    ]

    print("Archiving model into dlrm.mar")
    os.system(" ".join(cmd))
    print("Done")


if __name__ == "__main__":
    main()
