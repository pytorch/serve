import orjson
import requests

response = requests.post(
    "http://localhost:8080/predictions/llama-2-13b",
    data="Today the weather is really nice and I am planning on ",
    stream=True,
)

for chunk in response.iter_content(chunk_size=None):
    if chunk:
        data = chunk.decode("utf-8")
        data = orjson.loads(data)
        print(data["text"], end=" ", flush=True)

print("")
