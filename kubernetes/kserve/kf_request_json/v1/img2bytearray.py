""" Converts an image to bytesarray """
import base64
import json
import argparse
from io import BufferedReader
from typing import Dict, List

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="converts image to bytes array",
                    type=str)
args: argparse.Namespace = parser.parse_args()

image: BufferedReader = open(args.filename, 'rb') #open binary file in read mode
image_read: bytes = image.read()
image_64_encode: bytes = base64.b64encode(image_read)
bytes_array: str = image_64_encode.decode('utf-8')
request: Dict[str, List[Dict[str, str]]] = {
  "instances":[
    {
      "data": bytes_array
    }
  ]
}

with open('input.json', 'w') as outfile:
  json.dump(request, outfile, indent=4, sort_keys=True)
