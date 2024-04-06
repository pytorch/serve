import json

import requests

data = {
    "prompt": "What is the difference between cricket and baseball?",
    "params": {"temperature": 0.5, "top_p": 0.95, "max_new_tokens": 50},
}

response = requests.post(
    "http://localhost:8080/predictions/mixtral-moe",
    data=json.dumps(data),
    stream=True,
)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        data = chunk.decode("utf-8")
        print(data, end="", flush=True)

print("")
