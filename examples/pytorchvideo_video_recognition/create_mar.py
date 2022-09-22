"""
This script creates a DLRM model and packs it into a TorchServe mar file
"""

import os

import torch

MODEL_PT_FILE = "slowfast_r50.pt"


def create_pt_file(output_file: str) -> None:

    # model = SlowFast()
    model_name = "slowfast_r50"
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", model=model_name, pretrained=True
    )

    torch.save(model.cpu().state_dict(), output_file)


def main():
    print(f"Creating SlowFast_R50 model and saving state_dict to {MODEL_PT_FILE}")
    create_pt_file(MODEL_PT_FILE)

    cmd = [
        "torch-model-archiver",
        "--model-name slowfast_r50",
        "--version 1.0",
        "--model-file slowfast_r50_model.py",
        f"--serialized-file {MODEL_PT_FILE}",
        "--handler video_handler.py",
        "--extra-files transform_config.py,kinetics_classnames.json",
        "--force",
    ]

    print("Archiving model into slowfast_r50.mar")
    os.system(" ".join(cmd))
    print("Done")


if __name__ == "__main__":
    main()
