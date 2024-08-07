import requests

prompt = "What's new with Llama 3.1?"

RAG_EP = "http://localhost:8080/predictions/rag"

response = requests.post(url=RAG_EP, data=prompt)

response = requests.post(
    url="http://localhost:8080/predictions/llama3-8b-instruct",
    data=response.text.encode("utf-8"),
)
print(f"Question: {prompt}")
print(f"Answer: {response.text}")
