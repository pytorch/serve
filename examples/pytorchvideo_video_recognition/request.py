""" preprocessing steps to select a video
clip from the full video needs to be done
to make the video ready for request.
"""
import json

import requests

with open("kinetics_classnames.json") as jsonfile:
    info = json.load(jsonfile)

files = {
    "data": open("archery.mp4", "rb"),
}
response = requests.post("http://localhost:8080/predictions/slowfast_r50", files=files)
data = response.content.decode("UTF-8")
print(data)
