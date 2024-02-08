import argparse

import torch
import torch._export
from model import ModelArgs, Transformer


def load_checkpoint(checkpoint):
    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, map_location="cpu")
    gptconf = ModelArgs(**checkpoint_dict["model_args"])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, gptconf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", type=str, default="llama2.so", help="the output filepath"
    )
    parser.add_argument("--checkpoint", type=str, help="checkpoint .pt")
    args = parser.parse_args()
    model, config = load_checkpoint(args.checkpoint)
    x = torch.randint(0, config.vocab_size, (1, config.max_seq_len // 2))
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=config.max_seq_len)
    torch._C._GLIBCXX_USE_CXX11_ABI = True
    so_path = torch._export.aot_compile(
        model,
        (x,),
        dynamic_shapes={"tokens": (None, seq_len_dim)},
        options={"aot_inductor.output_path": args.filepath},
    )
