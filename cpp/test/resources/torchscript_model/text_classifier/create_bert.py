import pathlib
import time
from argparse import ArgumentParser
from typing import Any, List

import requests
import torch
import torch.nn as nn
import torchtext.functional as F
from torch import Tensor
from torchtext import transforms
from transformers import AutoModelForSequenceClassification


class CombinedModel(nn.Module):
    def __init__(self, tokenizer: nn.Module, model: nn.Module):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._seq_start = 101
        self._seq_end = 102
        self._dummy_param = nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _forward(self, batch_1: List[str], batch_2: List[str]) -> Tensor:
        tokens_1: List[List[str]] = []
        tokens_2: List[List[str]] = []

        tokens_1 = self._tokenizer._batch_encode(batch_1)
        tokens_2 = self._tokenizer._batch_encode(batch_2)

        lengths_1 = [len(t) for t in tokens_1]
        lengths_2 = [len(t) for t in tokens_2]

        max_length = max([l1 + l2 for l1, l2 in zip(lengths_1, lengths_2)])

        tokens_1 = [[int(i) for i in o] for o in tokens_1]
        tokens_2 = [[int(i) for i in o] for o in tokens_2]

        tokens_1 = [[self._seq_start] + s + [self._seq_end] for s in tokens_1]
        tokens_2 = [s + [self._seq_end] for s in tokens_2]

        tokens = [s1 + s2 for s1, s2 in zip(tokens_1, tokens_2)]
        tokens = F.to_tensor(tokens, padding_value=0).to(self._dummy_param.device)

        ids = [
            (l1 + 2)
            * [
                0,
            ]
            + (l2 + 1)
            * [
                1,
            ]
            for l1, l2 in zip(lengths_1, lengths_2)
        ]
        masks = [
            len(i)
            * [
                1,
            ]
            for i in ids
        ]

        ids = F.to_tensor(ids, padding_value=0).to(self._dummy_param.device)

        masks = F.to_tensor(masks, padding_value=0).to(self._dummy_param.device)

        return torch.softmax(
            self._model(input_ids=tokens, token_type_ids=ids, attention_mask=masks)[0],
            1,
        )

    def forward(self, batch_1: Any, batch_2: Any) -> Tensor:
        if torch.jit.isinstance(batch_1, str) and torch.jit.isinstance(batch_2, str):
            return self._forward([batch_1], [batch_2])
        if torch.jit.isinstance(batch_1, List[str]) and torch.jit.isinstance(
            batch_2, List[str]
        ):
            return self._forward(batch_1, batch_2)
        else:
            raise TypeError("Input type not supported")


def main(args):
    tokenizer_file = args.tokenizer_file

    LOCAL_FILE = str(pathlib.Path(__file__).parent / "vocab.txt")
    if not pathlib.Path(LOCAL_FILE).exists():
        VOCAB_FILE = (
            "https://huggingface.co/bert-base-cased-finetuned-mrpc/raw/main/vocab.txt"
        )
        response = requests.get(VOCAB_FILE)
        with open(LOCAL_FILE, "wb") as f:
            f.write(response.content)

    tokenizer = transforms.BERTTokenizer(vocab_path=LOCAL_FILE, do_lower_case=False)

    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    @torch.no_grad()
    def run_model(model_, s0, s1):
        st = time.time()
        classification_logits = model_(s0, s1)
        paraphrase_results = classification_logits.cpu().tolist()

        if args.verbose:
            print(f"Execution time: {1000*(time.time() - st):.2f} ms")
            print(
                "\n".join(
                    [f"{round(p[1] * 100)}% paraphrase" for p in paraphrase_results]
                )
            )

        return classification_logits

    bert = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased-finetuned-mrpc", torchscript=True
    )

    max_sequence_length = 64

    dummy_input = [
        torch.zeros([1, max_sequence_length], dtype=torch.long, device=bert.device),
        torch.zeros([1, max_sequence_length], dtype=torch.long, device=bert.device),
        torch.ones([1, max_sequence_length], dtype=torch.long, device=bert.device),
    ]

    bert = torch.jit.trace(bert, dummy_input)

    model = CombinedModel(tokenizer, bert)

    model.eval()

    logits_eager = run_model(
        model,
        [sequence_0, sequence_1, sequence_2],
        [sequence_1, sequence_2, sequence_0],
    )

    jit_model = torch.jit.script(model)
    logits_script = run_model(
        jit_model,
        [sequence_0, sequence_1, sequence_2],
        [sequence_1, sequence_2, sequence_0],
    )
    torch.jit.save(jit_model, tokenizer_file)

    loaded_jit_model = torch.jit.load(tokenizer_file)
    logits_loaded = run_model(
        loaded_jit_model,
        [sequence_0, sequence_1, sequence_2],
        [sequence_1, sequence_2, sequence_0],
    )

    assert torch.allclose(logits_eager, logits_script)
    assert torch.allclose(logits_eager, logits_loaded)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-file", default="tokenizer.pt", type=str)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
