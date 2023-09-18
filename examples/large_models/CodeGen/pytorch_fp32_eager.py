import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# generate args
max_length = 128
generate_kwargs = dict(max_length=max_length)

# load model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi")
model = model.eval()

############ benchmark ############
# input prompt
prompt = "# This Python script demonstrates a basic Multi-Layer Perceptron (MLP) model for image classification. Using PyTorch machine-learning framework library, it defines a simple MLP architecture, loads the datasets, preprocesses the input images, postprocesses the outputs, and trains it on the training data images. Finally, it evaluates the model's performance on the evaluation data images."
input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size) # 80

# start
total_time = 0.0
num_iter = 100
num_warmup = 10
total_list = []
with torch.inference_mode(), torch.no_grad():
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_ids = model.generate(input_ids, **generate_kwargs)
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        toc = time.time()
        
        print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
        if i >= num_warmup:
            total_time += toc - tic
            

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
###################################