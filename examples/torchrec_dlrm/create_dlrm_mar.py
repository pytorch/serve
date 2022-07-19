"""
This script creates a DLRM model and packs it into a TorchServe mar file
"""

import os

from dlrm_factory import DLRMFactory

MODEL_PT_FILE = "dlrm.pt"


def main():
    module = DLRMFactory()

    # torch.save(module.cpu().state_dict(), MODEL_PT_FILE)

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
