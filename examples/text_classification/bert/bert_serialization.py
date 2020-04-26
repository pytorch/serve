from transformers import BertTokenizer, BertForSequenceClassification
import torch

""" This script serialize a Bert model off-the-shelf for a sequence classifcation with
four output classes, a fine-tune process can be added to this before serializing the model. """

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1

traced_model = torch.jit.trace(model, [input_ids])
torch.jit.save(traced_model, "traced_bert.pt")
