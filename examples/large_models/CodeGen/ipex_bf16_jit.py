import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# generate args
num_beams = 4
max_length = 128
batch_size = 1
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams, max_length=max_length)

############ IPEX ############
# import ipex
import intel_extension_for_pytorch as ipex
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

# jit
torch._C._jit_set_texpr_fuser_enabled(False)

# device
device = torch.device("cpu")

# dtype
amp_enabled = True
amp_dtype = torch.bfloat16

# load model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi", torchscript=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi", torchscript=True, trust_remote_code=True)

model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# to ipex
model = ipex._optimize_transformers(model.eval(), dtype=amp_dtype, inplace=True)

# dummy past key values
beam_idx_tmp = torch.zeros(
    (2048, int(batch_size * num_beams)), dtype=torch.long
).contiguous()
past_key_values = tuple(
    [
        (
            torch.zeros([1, 1, 1, 1]).contiguous(),
            torch.zeros([1, 1, 1, 1]).contiguous(),
            beam_idx_tmp,
            torch.zeros(1, dtype=torch.long).contiguous(),
        )
        for i in range(model.config.n_layer)
    ]
)

if not hasattr(model, "trace_graph"): 
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids))
    position_ids = torch.arange(len(input_ids))

    example_inputs = {
        "input_ids": input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "past_key_values": past_key_values,
    }
    
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type="cpu",
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        trace_model = torch.jit.trace(
            model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False
        )
        trace_model = torch.jit.freeze(trace_model)
        setattr(model, "trace_graph", trace_model)
#############################

############ benchmark ############
# input prompt
prompt = "# This Python script demonstrates a basic Multi-Layer Perceptron (MLP) model for image classification. Using PyTorch machine-learning framework library, it defines a simple MLP architecture, loads the datasets, preprocesses the input images, postprocesses the outputs, and trains it on the training data images. Finally, it evaluates the model's performance on the evaluation data images."
input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size) #80

# start
total_time = 0.0
num_iter = 100
num_warmup = 10
total_list = []
with torch.inference_mode(), torch.no_grad(), torch.autocast(
    device_type="cpu",
    enabled=amp_enabled,
    dtype=amp_dtype if amp_enabled else None,
):
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