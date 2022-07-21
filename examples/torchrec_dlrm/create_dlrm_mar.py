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
    create_pt_file(MODEL_PT_FILE)

    cmd = [
        "torch-model-archiver",
        "--model-name dlrm",
        "--version 1.0",
        f"--serialized-file {MODEL_PT_FILE}",
        "--model-file dlrm_factory.py",
        "--handler dlrm_handler.py",
        "--force",
    ]

    os.system(" ".join(cmd))


if __name__ == "__main__":
    main()
