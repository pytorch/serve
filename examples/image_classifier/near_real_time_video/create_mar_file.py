"""
This script downloads the Resnet-18 model and packs it into a TorchServe mar file
"""

import argparse
import os
import shutil

MODEL_PTH_FILE = "resnet18-f37072fd.pth"
MODEL_STORE = "model_store"
MAR_FILE = "resnet-18.mar"


def download_pth_file(output_file: str) -> None:
    if not os.path.exists(output_file):
        cmd = ["wget", " https://download.pytorch.org/models/resnet18-f37072fd.pth"]
        print("Downloading resnet-18 pth file")
        os.system(" ".join(cmd))


def create_mar():

    if args.client_batching:
        cmd = [
            "torch-model-archiver",
            "--model-name resnet-18",
            "--version 1.0",
            f"--serialized-file {MODEL_PTH_FILE}",
            "--model-file examples/image_classifier/near_real_time_video/model.py",
            "--extra-files examples/image_classifier/index_to_name.json",
            "--handler examples/image_classifier/near_real_time_video/near_real_time_video_handler.py",
            "--force",
        ]
    else:
        cmd = [
            "torch-model-archiver",
            "--model-name resnet-18",
            "--version 1.0",
            f"--serialized-file {MODEL_PTH_FILE}",
            "--model-file examples/image_classifier/near_real_time_video/model.py",
            "--extra-files examples/image_classifier/index_to_name.json",
            "--handler image_classifier",
            "--force",
        ]

    print(f"Archiving resnet-18 model into {MAR_FILE}")
    os.system(" ".join(cmd))


def move_mar_file():
    if not os.path.exists(MODEL_STORE):
        os.makedirs(MODEL_STORE)

    shutil.move(MAR_FILE, os.path.join(MODEL_STORE, MAR_FILE))
    print(f"Moving {MAR_FILE} into {MODEL_STORE}")


def main():
    download_pth_file(MODEL_PTH_FILE)
    create_mar()
    move_mar_file()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-batching",
        help="To use client side batching methodology",
        action="store_true",
    )
    args = parser.parse_args()

    main()
