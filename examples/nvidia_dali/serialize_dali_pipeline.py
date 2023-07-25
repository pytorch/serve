import nvidia.dali as dali
import nvidia.dali.types as types
import yaml


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model-config.yaml")
    return parser.parse_args()


@dali.pipeline_def
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.html
def pipe():
    jpegs = dali.fn.external_source(dtype=types.UINT8, name="source")
    decoded = dali.fn.decoders.image(jpegs, device="mixed", output_type=types.GRAY)
    normalized = dali.fn.crop_mirror_normalize(
        decoded,
        mean=[0.1307 * 255],
        std=[0.3081 * 255],
    )
    return normalized


def main():
    config = {}
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    batch_size = config["dali"]["batch_size"]
    num_threads = config["dali"]["num_threads"]
    device_id = config["dali"]["device_id"]
    seed = config["dali"]["seed"]
    pipeline_filename = config["dali"]["pipeline_file"]

    pipeline = pipe(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed
    )
    pipeline.serialize(filename=pipeline_filename)
    print("Saved {}".format(pipeline_filename))


if __name__ == "__main__":
    args = parse_args()
    main()
