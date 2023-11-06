import argparse

import mii

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-m", type=str, required=True, help="Model Name")
parser.add_argument(
    "--prompt", "-p", type=str, required=True, help="Input Prompt for text generation"
)
parser.add_argument(
    "--tensor_parallel",
    "-tp",
    type=int,
    default=1,
    required=False,
    help="tensor parallel degree",
)
parser.add_argument(
    "--dtype", "-dtype", type=str, default="fp16", required=False, help="quantization"
)
args = parser.parse_args()

model_config = {"tensor_parallel": args.tensor_parallel, "dtype": args.dtype}


pipe = mii.pipeline(
    model_name_or_path=args.model_path,
    model_config=model_config,
)

output = pipe(args.prompt)
