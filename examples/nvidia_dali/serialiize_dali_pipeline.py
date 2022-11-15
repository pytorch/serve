import json
import os

import nvidia.dali as dali
import nvidia.dali.types as types


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="./model.dali")
    parser.add_argument("--config", default="dali_config.json")
    return parser.parse_args()


@dali.pipeline_def
def pipe():
    jpegs = dali.fn.external_source(dtype=types.UINT8, name="my_source")
    decoded = dali.fn.decoders.image(jpegs, device="mixed")
    resized = dali.fn.resize(
        decoded,
        size=[256],
        subpixel_scale=False,
        interp_type=types.DALIInterpType.INTERP_LINEAR,
        antialias=True,
        mode="not_smaller",
    )
    normalized = dali.fn.crop_mirror_normalize(
        resized,
        crop_pos_x=0.5,
        crop_pos_y=0.5,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return normalized


def main(filename):
    with open(args.config) as fp:
        config = json.load(fp)
    batch_size = config["batch_size"]
    num_threads = config["num_threads"]
    device_id = config["device_id"]

    pipe1 = pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    pipe1.serialize(filename=filename)
    print("Saved {}".format(filename))


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    main(args.save)
