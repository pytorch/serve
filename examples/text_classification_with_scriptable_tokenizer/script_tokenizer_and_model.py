"""
Combine tokenizer and XLM-RoBERTa model pretrained on SST-2 Binary text classification
"""

import argparse
from typing import Any

import torch
import torchtext.functional as F
import torchtext.transforms as T
from torch import nn
from torch.hub import load_state_dict_from_url
from torchtext.models import XLMR_BASE_ENCODER, RobertaClassificationHead

PADDING_IDX = 1
BOS_IDX = 0
EOS_IDX = 2
MAX_SEQ_LEN = 256
# Vocab file for the pretrained XLM-RoBERTa model
XLMR_VOCAB_PATH = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
# Model file for ther pretrained SentencePiece tokenizer
XLMR_SPM_MODEL_PATH = (
    r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
)


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
            x (Any): Tokenizer output. As we script the combined model, we need to
            hint the type of the input argument of the adapter module which TorchScript
            identified as Any. Chosing a more restrictive type lets the scripting fail.

        Returns:
            (Tensor): On device text tensor with batch dimension
        """
        tokens = F.to_tensor(tokens, padding_value=self._padding_idx).to(
            self._dummy_param.device
        )
        # If a single sample is tokenized we need to add the batch dimension
        if len(tokens.shape) < 2:
            return tokens.unsqueeze(0)
        return tokens


def main(args):

    # Chain preprocessing steps as defined during training.
    text_transform = T.Sequential(
        T.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH),
        T.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH)),
        T.Truncate(MAX_SEQ_LEN - 2),
        T.AddToken(token=BOS_IDX, begin=True),
        T.AddToken(token=EOS_IDX, begin=False),
    )

    NUM_CLASSES = 2
    INPUT_DIM = 768

    classifier_head = RobertaClassificationHead(
        num_classes=NUM_CLASSES, input_dim=INPUT_DIM
    )

    model = XLMR_BASE_ENCODER.get_model(head=classifier_head)

    # Load trained parameters and load them into the model
    model.load_state_dict(torch.load(args.input_file))

    # Chain the tokenizer, the adapter and the model
    combi_model = T.Sequential(
        text_transform,
        TokenizerModelAdapter(PADDING_IDX),
        model,
    )

    combi_model.eval()

    # Make sure to move the model to CPU to avoid placement error during loading
    combi_model.to("cpu")

    combi_model_jit = torch.jit.script(combi_model)

    torch.jit.save(combi_model_jit, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine tokenzier and model.")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)

    args = parser.parse_args()
    main(args)
