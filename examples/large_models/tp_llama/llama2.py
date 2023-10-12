# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import math
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from torch.distributed._tensor.placement_types import Replicate, Shard
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp

import torch.nn.functional as F
from checkpoint_converter import build_distributed_state_dict_from_consolidated
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from llama2_tokenizer import Tokenizer
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )

from copy import deepcopy
from dataclasses import dataclass, asdict, fields
current_working_directory = os.getcwd()
sys.path.insert(0,current_working_directory)

log = logging.getLogger(__name__)

def dataclass_to_json(dc):
    return json.dumps(asdict(dc))

def json_to_dataclass(json_str, dataclass_type):
    data = json.loads(json_str)
    return dataclass_type(**data)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


# TODO: update this to use RMSNorm in MultiModal
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
   
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )



class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        dim: int,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        tp_degree = int(os.environ["WORLD_SIZE"])
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads//tp_degree
        self.n_local_kv_heads = self.n_kv_heads//tp_degree
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._init_cache_k()
        self._init_cache_v()

    def _init_cache_k(self):
        self.cache_k = torch.zeros(
            (
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def _init_cache_v(self):
        self.cache_v = torch.zeros(
            (
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
       
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        #calling PT SDPA to enable using Flash Attention 2 and Xformer memory efficient kernels.

        output = torch.nn.functional.scaled_dot_product_attention(xq.transpose(1,2), keys.transpose(1,2), values.transpose(1,2), attn_mask=mask, dropout_p=0.0, is_causal=False)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        a= self.w1(x)
        b =F.silu(a)
        c= self.w3(x)
        return self.w2(b*c)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        n_heads: int,
        n_kv_heads: int,
        dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(
            n_heads, n_kv_heads, dim, max_batch_size, max_seq_len
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
   
class Transformer(nn.Module):
    """
    LLama2 implementation, free of any coupling to parallelism implementations, heavily drawn from
    https://github.com/facebookresearch/llama.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id,
                    n_heads,
                    n_kv_heads,
                    dim,
                    multiple_of,
                    ffn_dim_multiplier,
                    max_batch_size,
                    max_seq_len,
                    norm_eps,
                )
            )

        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads, self.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, padding: torch.Tensor=None):
        bsz, seqlen = tokens.shape
        # print(
        #     f"RV: before embedding lookup, input {tokens}, start:{start_pos}",
        #     flush=True,
        # )
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), torch.finfo(h.dtype).min, device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
            if padding is not None:
                assert padding.size(0) == bsz
                mask = mask.expand(bsz, 1, seqlen, seqlen).clone()
                mask_cond = torch.arange(mask.size(-1), device=mask.device)
                mask.masked_fill_(mask_cond < (padding).view(-1,1,1,1), torch.finfo(h.dtype).min)
        elif padding is not None:
            seqlen_with_past = seqlen + start_pos
            mask = torch.full(
                (bsz, 1, 1, seqlen_with_past), torch.finfo(h.dtype).min, device=tokens.device
            )
            mask_cond = torch.arange(mask.size(-1), device=mask.device)
            mask.masked_fill_(mask_cond + 1 > padding.view(-1,1,1,1), 0)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    
    @torch.no_grad()
    def reset_parameters(self):
       for layer in self.layers:
           for submodule in layer.modules():
               for param_name, param in submodule.named_parameters(recurse=False):
                    if param.is_meta:
                        materialized_param = nn.Parameter(
                            torch.empty_like(param, dtype=torch.bfloat16, device=torch.device("cuda"))
                        )
                        nn.init.uniform_after(materialized_param)
                        setattr(submodule, param_name, materialized_param)


### --- Utilities for model creation / loading ---- ####


def _build_model_args(ckpt_dir: str, max_seq_len, max_batch_size) -> ModelArgs:
    """
    Reads params.json from checkpoint and builds ModelArgs to initialize
    model with.
    """
    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.loads(f.read())

    # Some checkpoints have other details besides "model", fix this up and use a
    # clearly specified format.
    model_params = params.get("model", params)
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dim=model_params["dim"],
        n_layers=model_params["n_layers"],
        n_heads=model_params["n_heads"],
        n_kv_heads=model_params.get("n_kv_heads", model_params["n_heads"]),
        multiple_of=model_params["multiple_of"],
        ffn_dim_multiplier=model_params.get("ffn_dim_multiplier", None),
        norm_eps=model_params["norm_eps"],
    )
    return model_args


def _create_tokenizer(tokenizer_path: str) -> Tokenizer:
    local_tokenizer_path = tokenizer_path
    log.debug(f"successfully saved tokenizer to {local_tokenizer_path}")
    tokenizer = Tokenizer(model_path=local_tokenizer_path)
    return tokenizer


def _init_local_model(model_args: ModelArgs) -> Transformer:
    with torch.device("meta"):
        model = Transformer(
            model_args.vocab_size,
            model_args.n_layers,
            model_args.dim,
            model_args.n_heads,
            model_args.n_kv_heads,  # pyre-ignore[6]
            model_args.multiple_of,
            model_args.ffn_dim_multiplier,
            model_args.max_batch_size,
            model_args.max_seq_len,
            model_args.norm_eps,
        )

    model.freqs_cis = precompute_freqs_cis(
        model.dim // model.n_heads, model.max_seq_len * 2
    )
    for tformer_block in model.layers:
        tformer_block.attention._init_cache_k()
        tformer_block.attention._init_cache_v()

    return model


def get_consolidated_ckpt_path(
    ckpt_dir: Union[str, Path], mp_rank: int = 0, mp_size: int = 1
) -> Union[str, Path]:
  
    if mp_size == 1:
        assert mp_rank == 0
        filename = "consolidated.00.pth"
    else:
        filename = f"consolidated.{mp_rank:02d}.pth"
    if isinstance(ckpt_dir, Path):
        return ckpt_dir / filename
    else:
        return os.path.join(ckpt_dir, filename)

def _convert_fairscale_checkpoints(meta_model, model_parallel_size: int, original_ckpt_dir: str, save_checkpoint_dir: str):
    mp_group, _ = dist.new_subgroups(group_size=model_parallel_size)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise RuntimeError("Expected local_rank to be set, but it is not!")
    mp_rank = local_rank % model_parallel_size
    
    state_dict_pth = get_consolidated_ckpt_path(
        ckpt_dir=original_ckpt_dir, mp_rank=mp_rank, mp_size=model_parallel_size
    )
    state_dict = torch.load(state_dict_pth)
    dist_state_dict = build_distributed_state_dict_from_consolidated(
        meta_model, state_dict, model_parallel_world_size=model_parallel_size,use_dtensor=True
    )
    dist_cp.save_state_dict(
            state_dict=dist_state_dict,
            storage_writer=dist_cp.FileSystemWriter(save_checkpoint_dir),
        )
    

def _load_checkpoint(mesh, model, meta_model, model_parallel_size: int, ckpt_dir: str) -> None:
    mp_group, _ = dist.new_subgroups(group_size=model_parallel_size)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise RuntimeError("Expected local_rank to be set, but it is not!")
    mp_rank = local_rank % model_parallel_size
    state_dict_pth = get_consolidated_ckpt_path(
        ckpt_dir=ckpt_dir, mp_rank=mp_rank, mp_size=model_parallel_size
    )
    state_dict = torch.load(state_dict_pth)
    dist_state_dict = build_distributed_state_dict_from_consolidated(
        meta_model, state_dict, model_parallel_world_size=model_parallel_size,use_dtensor=True
    )
    CHECKPOINT_DIR="converted_checkpoints"
    dist_cp.save_state_dict(
            state_dict=dist_state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )
    
    converting_Dtensor_to_tensor(model,dist_state_dict,mesh )
    check_dtensor(dist_state_dict)
    log.debug("build distributed_state_dict")
    missing_keys, unexpected_keys = model.load_state_dict(dist_state_dict, strict=False)
    assert not missing_keys
    assert len(unexpected_keys) == 1 and "freqs" in unexpected_keys[0]

def _load_tp_checkpoints(tp_model,CHECKPOINT_DIR):
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise RuntimeError("Expected local_rank to be set, but it is not!")
    tp_state_dict = tp_model.state_dict()
    dist_cp.load_state_dict(
            state_dict=tp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
    tp_model.load_state_dict(tp_state_dict)
    
def parallelize_llama_MLP_block(model, module_path, mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=mesh,
        parallelize_plan={
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        },
        # tp_mesh_dim=0,
    )
    return parallelized_block

def parallelize_llama_attn_block(model, module_path, twod_mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=twod_mesh,
        parallelize_plan={
            "wq": ColwiseParallel(),
            "wk": ColwiseParallel(),
            "wv": ColwiseParallel(),
            "wo": RowwiseParallel(),
        },
        # tp_mesh_dim=0,
    )
    return parallelized_block

def tp_llama(model, mesh):
    for i in range(model.n_layers):
        # print(f" i number of layers {i}*********************")
        block = parallelize_llama_MLP_block(model, f"layers.{i}.feed_forward", mesh)
        block = parallelize_llama_attn_block(model, f"layers.{i}.attention", mesh)
        
def print_submodules(model):
        for name, module in model.named_modules():
            print(f"Module name: {name}")
            # print(module)
            print()       

def check_dtensor(state_dict):
    for fqn, tensor in state_dict.items():
        try:
            is_dtensor = isinstance(tensor, DTensor)
        except:
            is_dtensor = False
            
        print(f"The model FQN: {fqn}, is DTensor {is_dtensor}")
        
def converting_Dtensor_to_tensor(model_tp, dist_state_dict, mesh):          
# Make sure this covers all non DTensor FQNs.
    # model is the tp_model
    for fqn in model_tp.state_dict():
        if not isinstance(model_tp.state_dict()[fqn], DTensor):
        # #     # Convert dist_state_dict[fqn] into non-DTensor
            
            if isinstance(dist_state_dict[fqn], DTensor):
                # Not sure best way to materialize full DTensor on each rank Doing it by
                # redistributing it to a world_size = 1 DeviceMesh, and then to_local.
                unsharded_dt = dist_state_dict[fqn].redistribute(device_mesh=mesh, placements=[Replicate()])
                dist_state_dict[fqn] = unsharded_dt.to_local()
                
class Llama:
    @staticmethod
    def build(
        model_args: str,
        converted_ckpt_dir:str,
        tokenizer_path: str,
    ) -> "Llama":
        """
        Heavily motivated from https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L51,
        and adapted for native parallelism APIs.
        """
        start = time.time()
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            raise RuntimeError("Expected local_rank to be set, but it is not!")

        torch.cuda.set_device(local_rank)
        # model_args = _build_model_args(ckpt_dir, max_seq_len, max_batch_size)
        # file_path = os.path.join(converted_ckpt_dir, 'model_args.json')

        with open(model_args, 'r') as file:
          loaded_json = file.read()

        model_args = json_to_dataclass(loaded_json, ModelArgs)
        
        tokenizer = _create_tokenizer(tokenizer_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        model = _init_local_model(model_args)
        mesh = (
        DeviceMesh(
            device_type="cuda",
            mesh=list(range(dist.get_world_size())),
        ))
        
        tp_llama(model, mesh)
        
        model.to_empty(device='cuda')
        model.reset_parameters()
        log.debug(f"Rank {dist.get_rank()}: created FSDP model {model}")

        _load_tp_checkpoints(model,converted_ckpt_dir)
        param_numel = sum(p.numel() for p in model.parameters())
        log.debug(
            f"Loaded {param_numel * dist.get_world_size()} params (across all workers) in {time.time() - start:.2f} seconds"
        )
        return Llama(model, tokenizer)

    @staticmethod
    def convert_checkpoints(
        original_ckpt_dir: str,
        save_checkpoint_dir:str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: int,
    ) -> "Llama":
        """
        Heavily motivated from https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L51,
        and adapted for native parallelism APIs.
        """
        start = time.time()
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank == -1:
            raise RuntimeError("Expected local_rank to be set, but it is not!")

        torch.cuda.set_device(local_rank)
        model_args = _build_model_args(original_ckpt_dir, max_seq_len, max_batch_size)
        tokenizer = _create_tokenizer(tokenizer_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        json_args = dataclass_to_json(model_args)
        with open('model_args.json', 'w') as file:
            file.write(json_args)
            
        model = _init_local_model(model_args)
        
        _convert_fairscale_checkpoints(model, model_parallel_size=model_parallel_size, original_ckpt_dir=original_ckpt_dir, save_checkpoint_dir=save_checkpoint_dir)
    
        log.debug(
            f"the  checkpoints have been converted to PTD compliant checkpoint and saved in {save_checkpoint_dir}"
        )

    
    def __init__(self, model: Union[FSDP, Transformer], tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer