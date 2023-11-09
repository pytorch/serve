import argparse

import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-m", type=str, required=True, help="Model Name")
parser.add_argument(
    "--prompt", "-p", type=str, required=True, help="Input Prompt for text generation"
)

args = parser.parse_args()

pipe = mii.pipeline(
    model_name_or_path=args.model_path,
)

output = pipe(args.prompt)
