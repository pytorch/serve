import argparse
import json
from datetime import datetime

import numpy as np
import requests
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--url", type=str, required=True, help="Torchserve inference endpoint"
)
parser.add_argument(
    "--prompt", type=str, required=True, help="Prompt for image generation"
)
parser.add_argument(
    "--filename",
    type=str,
    default="output-{}.jpg".format(str(datetime.now().strftime("%Y%m%d%H%M%S"))),
    help="Filename of output image",
)
args = parser.parse_args()

response = requests.post(args.url, data=args.prompt)
# Contruct image from response
image = Image.fromarray(np.array(json.loads(response.text), dtype="uint8"))
image.save(args.filename)
