import requests

prompt = "What's new with Llama 3.1?"

response = requests.post(url="http://localhost:8080/predictions/rag", data=prompt)
print(response.text)
