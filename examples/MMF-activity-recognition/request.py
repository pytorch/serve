import requests
import json
''' preprocessing steps to select a video
clip from the full video needs to be done
to make the video ready for request.
'''
with open("372CC.info.json") as jsonfile:
    info = json.load(jsonfile)

files = {
        'data': open('372CC.mp4','rb'),
        'script': info['script'],
        'labels':info['action_labels']
        }
response = requests.post('http://localhost:8080/predictions/MMF_activity_recognition',
 files=files)
data = response.content
with open("response.txt", "wb") as response_handler:
    response_handler.write(data)
