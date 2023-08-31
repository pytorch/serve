import requests

url = "http://127.0.0.1:8080/predictions/rnnt"

with open('1089-134686-0000.wav', 'rb') as f:
    bytes = f.read()

r = requests.post(url, data = bytes)

print(r.text)