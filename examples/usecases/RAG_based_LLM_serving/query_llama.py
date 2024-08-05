import requests

prompt = "What's new with Llama 3.1?"


response = requests.post(
    url="http://localhost:8080/predictions/llama3-8b-instruct", data=prompt
)
print(f"Question: {prompt}")
print(f"Answer: {response.text}")
