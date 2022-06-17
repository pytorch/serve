"""
SST-2 Binary text classification with XLM-RoBERTa model
"""

from typing import Any

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchtext.functional as F
import torchtext.transforms as T
from torchtext.datasets import SST2
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

PADDING_IDX = 1
BOS_IDX = 0
EOS_IDX = 2
MAX_SEQ_LEN = 256
XLMR_VOCAB_PATH = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
XLMR_SPM_MODEL_PATH = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH),
    T.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH)),
    T.Truncate(MAX_SEQ_LEN - 2),
    T.AddToken(token=BOS_IDX, begin=True),
    T.AddToken(token=EOS_IDX, begin=False),
)


BATCH_SIZE = 16

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")


train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
train_datapipe = train_datapipe.batch(BATCH_SIZE)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
dev_datapipe = dev_datapipe.batch(BATCH_SIZE)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)

NUM_CLASSES = 2
INPUT_DIM = 768

classifier_head = RobertaClassificationHead(num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)

LEARNING_RATE = 1e-5
optim = AdamW(model.parameters(), lr=LEARNING_RATE)
criteria = nn.CrossEntropyLoss()


def train_step(input_tensor, target):
    """
    Performs a training step

    Args:
        input: Input data for step
        target: Target for step
    """
    output = model(input_tensor)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input_tensor, target):
    """
    Performs a evaluation step

    Args:
        input: Input data for step
        target: Target for step

    Return:
        (Tuple): Loss and accuracy
    """
    output = model(input_tensor)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    """
    Performs a evaluation

    Return:
        (Tuple): Loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input_tensor = F.to_tensor(batch["token_ids"], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input_tensor, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


def train():
    """
    Performs a training
    """
    num_epochs = 1

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_tensor = F.to_tensor(batch["token_ids"], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            train_step(input_tensor, target)

        loss, accuracy = evaluate()
        print(f"Epoch = [{epoch}], loss = [{loss}], accuracy = [{accuracy}]")


train()


class TokenizerModelAdapter(nn.Module):
    """
    TokenizerModelAdapter moves input onto device and adds batch dimension
    """
    def __init__(self, padding_idx):
        super().__init__()
        self._padding_idx = padding_idx
        self._dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, tokens: Any) -> torch.Tensor:
        """
        Moves input onto device and adds batch dimension.

        Args:
            x (Any): tokenizer output

        Returns:
            (Tensor): On device text tensor with batch dimension
        """
        tokens = F.to_tensor(tokens, padding_value=self._padding_idx).to(self._dummy_param.device)
        if len(tokens.shape) == 2:
            return tokens
        else:
            return tokens.unsqueeze(0)

combi_model = T.Sequential(
    text_transform,
    TokenizerModelAdapter(PADDING_IDX),
    model,
)

combi_model.eval()

combi_model.to("cpu")

combi_model_jit = torch.jit.script(combi_model)

torch.jit.save(combi_model_jit, "model.pt")
