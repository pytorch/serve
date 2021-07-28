import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import argparse

## Enable logging so we can see any important warnings
logger = logging.getLogger('Neuron')
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument(
"--batch-size",
action="store",
help="Supply a .yaml file with test_name, instance_id, and key_filename to re-use already-running instances",
)

arguments = parser.parse_args()

batch_size = int(arguments.batch_size)

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=128, pad_to_max_length=True, return_tensors="pt")
not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=128, pad_to_max_length=True, return_tensors="pt")

# Run the original PyTorch model on both example inputs
paraphrase_classification_logits = model(**paraphrase)[0]
not_paraphrase_classification_logits = model(**not_paraphrase)[0]

max_length=128
# Convert example inputs to a format that is compatible with TorchScript tracing
input_ids = paraphrase['input_ids']  # type:torch.Tensor
token_type_ids = paraphrase['token_type_ids']  # type:torch.Tensor
attention_mask = paraphrase['attention_mask']  # type:torch.Tensor
input_ids = input_ids.expand(batch_size, max_length)
token_type_ids = token_type_ids.expand(batch_size, max_length)
attention_mask = attention_mask.expand(batch_size, max_length)
example_inputs_paraphrase = input_ids, attention_mask, token_type_ids

input_ids = not_paraphrase['input_ids']  # type:torch.Tensor
token_type_ids = not_paraphrase['token_type_ids']  # type:torch.Tensor
attention_mask = not_paraphrase['attention_mask']  # type:torch.Tensor
input_ids = input_ids.expand(batch_size, max_length)
token_type_ids = token_type_ids.expand(batch_size, max_length)
attention_mask = attention_mask.expand(batch_size, max_length)
example_inputs_not_paraphrase = input_ids, attention_mask, token_type_ids

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron, using optimization level -O2
model_neuron = torch.neuron.trace(model, example_inputs_paraphrase, compiler_args=['-O2'])

# Verify the TorchScript works on both example inputs
paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)
not_paraphrase_classification_logits_neuron = model_neuron(*example_inputs_not_paraphrase)

# Save the TorchScript for later use
model_neuron.save(f"bert_neuron_{batch_size}.pt")