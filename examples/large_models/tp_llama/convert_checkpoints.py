import torch
from llama2 import Llama
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Tuple
import abc 
import fire


def convert_checkpoints(
    original_ckpt_dir: str,
    save_checkpoint_dir: str,
    tokenizer_path: str,
    model_parallel_size: int,
    max_seq_len: int=512,
    max_batch_size: int=4,
    ):
    dist.init_process_group("nccl")
    
    Llama.convert_checkpoints(
        original_ckpt_dir=original_ckpt_dir,
        save_checkpoint_dir=save_checkpoint_dir,
        tokenizer_path= tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
      
   
if __name__ == "__main__":
    fire.Fire(convert_checkpoints)
    