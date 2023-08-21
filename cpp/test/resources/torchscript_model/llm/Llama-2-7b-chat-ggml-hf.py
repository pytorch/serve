import requests
import json
import time

def send_text_file(url, file_path):
    with open(file_path, 'rb') as fp:
        file_bytes = fp.read()

    start_time = time.time()
    response = requests.post(url, data=file_bytes)
    time_taken = time.time() - start_time
    generated_answer = response.text
    print("Generated Anser: ", generated_answer)
    number_of_tokens = len(generated_answer.split(' '))
    print("Number of tokens: ", number_of_tokens)
    print("Time taken: ", time_taken)
    print("Tokens per second:", number_of_tokens / int(time_taken))


if __name__ == "__main__":
    url = "http://localhost:8080/predictions/llm"
    file_path = "llm_handler/prompt.txt"

    send_text_file(url, file_path)

