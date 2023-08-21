from transformers import AutoTokenizer
import transformers
import torch
import time

model = "meta-llama/Llama-2-7b-chat-hf"
hf_api_key = "<INSERT-YOUR-HF-KEY-HERE>"

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=hf_api_key)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=hf_api_key
)

start_time = time.time()
sequences = pipeline(
    'Hello my name is\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=512,
)
result = ""
for seq in sequences:
    result += seq['generated_text'] 
    print(f"Result: {seq['generated_text']}")
time_taken = time.time() - start_time

print("Generated String:", result)
print("Total time taken:", time_taken)

num_words = len(result.split(' '))

print("Total words generated: ", num_words)

tokens_per_second = num_words / int(time_taken)

print("Tokens per second: ", tokens_per_second)
