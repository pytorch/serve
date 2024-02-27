import torch
from transformers import AutoTokenizer

from ts.handler_utils.hf_batch_streamer import TextIteratorStreamerBatch


def test_hf_batch_streamer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    streamer = TextIteratorStreamerBatch(
        tokenizer=tokenizer, batch_size=2, skip_special_tokens=True
    )

    input1 = "hello world"
    input2 = "good day"

    for inputs in zip(tokenizer(input1)["input_ids"], tokenizer(input2)["input_ids"]):
        streamer.put(torch.tensor(inputs))

    streamer.end()

    output1 = ""
    output2 = ""

    for data in streamer:
        assert len(data) == 2
        output1 += data[0]["text"]
        output2 += data[1]["text"]

    assert output1 == input1
    assert output2 == input2
