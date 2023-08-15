import os

import nvidia.dali as dali
import nvidia.dali.types as types


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_threads", default=1)
    parser.add_argument("--device_id", default=1)
    parser.add_argument("--save", default="./default.dali")
    return parser.parse_args()


@dali.pipeline_def
def pipe():
    jpegs = dali.fn.external_source(dtype=types.UINT8, name="source")
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


def main(batch_size, num_threads, device_id, filename):
    pipeline = pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    pipeline.serialize(filename=filename)
    print(f"Saved {filename}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    main(args.batch_size, args.num_threads, args.device_id, args.save)
