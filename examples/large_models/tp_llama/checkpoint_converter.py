import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)


@dataclass
class FairScaleFSDPManagedParam:
    """
    Information about an original parameter managed by (fairscale) FSDP.
    Attributes:
        flat_param_key: FQN of the flat_param in FSDP this original param belongs to. This is the key in the fairscale state_dict.
        fqn: full original param FQN, with no FSDP prefixing, starting from root module.
        full_shape: full, unsharded parameter shape
        local_numels: numels value from the fairscale state_dict, unused for now
        data_tensor: Union[ShardedTensor, DTensor] - data tensor sharded in the PT-D style
    """

    flat_param_key: str  # this is the key in the fairscale state_dict
    fqn: str  # full FQN starting from root module, with no FSDP prefixing
    full_shape: torch.Size  # full, unsharded shape (original parameter shape)
    local_numels: int  # numels value from the fairscale state_dict
    data_tensor: torch.Tensor  # actual data tensor


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _verify_fqn_across_ranks(fqn, grp_gloo):
    olist = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(olist, fqn, group=grp_gloo)
    assert len(set(olist)) == 1
    assert olist[0] == fqn


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _all_gather_into_list(data_tensor, model_parallel_group):
    tensor_list = [
        torch.zeros_like(data_tensor).cuda()
        for _ in range(dist.get_world_size(model_parallel_group))
    ]
    dist.all_gather(tensor_list, data_tensor.cuda(), group=model_parallel_group)
    return tensor_list


def _is_tp_sharded(fqn: str) -> bool:
    """
    Returns whether a tensor given by the fqn is tensor parallel sharded.
    NOTE: this is currently done by inspection of the MF model and is quite
    brittle and would need to be updated if the MF sharding changes.
    """
    return (
        "attention" in fqn
        or "feed_forward" in fqn
        or "output" in fqn
        or "tok_embeddings" in fqn
    )
    
def _unshard_param(
    ref_state_dict,
    fqn,
    model_parallel_group,
    grp_gloo,
    data_tensor,
    tp_sharded_shape,
    ):
    """
    Unshards the row or col-wise sharded parameter.
    For rowwise, this is done by reshaping into the local shape, allgathering,
    and stacking rows. For colwise, the only difference is we stack columns.
    This is done via vstack and column_stack respectively.
    """
    mp_size = dist.get_world_size(model_parallel_group)
    # print(f"mp sizeeeeeeeeee {mp_size}")
    # print("-------------------------------")
    ref_shape = ref_state_dict[fqn].shape
    assert (
        ref_shape[0] == tp_sharded_shape[0] or ref_shape[1] == tp_sharded_shape[1]
    ), f"Expected sharded shape to match either row or col-wise, but does not: {ref_shape} {tp_sharded_shape}"
    _verify_fqn_across_ranks(fqn, grp_gloo)
    if ref_shape[0] != tp_sharded_shape[0]:
        assert ref_shape[0] == tp_sharded_shape[0] * mp_size
        # reshape the flat data_tensor into the rowwise shape
        data_tensor = data_tensor.reshape(tp_sharded_shape)
        # now, all_gather such tensors
        tensor_list = _all_gather_into_list(data_tensor, model_parallel_group)
        # stack rowwise to produce the final unsharded tensor
        data_tensor = torch.vstack(tensor_list).cpu()
        assert data_tensor.shape == ref_shape
        full_shape = data_tensor.shape
    elif (
        len(ref_shape) > 1
        and len(tp_sharded_shape) > 1
        and ref_shape[1] != tp_sharded_shape[1]
    ):
        assert ref_shape[1] == mp_size * tp_sharded_shape[1]
        # first, reshape the flat data_tensor into the colwise shape
        data_tensor = data_tensor.reshape(tp_sharded_shape)
        tensor_list = _all_gather_into_list(data_tensor, model_parallel_group)
        data_tensor = torch.column_stack(tensor_list).cpu()
        assert data_tensor.shape == ref_shape, f"{data_tensor.shape} vs {ref_shape}"
        full_shape = data_tensor.shape
    else:
        assert ref_shape == tp_sharded_shape  # not tensor parallel sharded
        full_shape = tp_sharded_shape
        logging.warning(f"{fqn} {ref_shape} {full_shape} - not sharded")
    return data_tensor, full_shape


def build_distributed_state_dict_from_consolidated(
    model: nn.Module,
    consolidated_state_dict: Dict[str, Tensor],
    model_parallel_world_size: int,
    offload_to_cpu: bool = False,
    use_dtensor: bool = False,
) -> Dict[str, Union[Tensor, DTensor, ShardedTensor]]:
    """
    Main API that takes a model (with no parallelism applied) and a fairscale checkpoint
    and builds a PT-D compliant distributed state dict. Note that this expects a consolidated
    checkpoint.

    Args:
        model (torch.nn.Module): module with no parallelism applied (i.e. result of `build_model` with parallel_impl=ParallelImpl.NONE)
        fs_state_dict (Dict[str, Any]): Fairscale consolidated
        offload_to_cpu (bool): Whether to offload the resulting state_dict to CPU (default: False)
        use_dtensor (bool): Whether to use PyTorch Distributed Tensor instead of ShardedTensor (default: False)
            (this will eventually default to True)
        model_parallel_world_size: Model parallel world size that was used to create the consolidated checkpoint.
            This can be obtained by checking the number of consolidated0x.pth files in the checkpoint directory.

    Example usage::
        ```
        
        MODEL_PARALLEL_SIZE = 8
        ckpt_path = get_consolidated_ckpt_path(
            ckpt_dir=PTH_65b, mp_rank=local_rank, mp_size=MODEL_PARALLEL_SIZE
        )
        state_dict = torch.load(ckpt_path)
        # Build a local LLaMA with no parallelism
        model = build_model(...)
        sharded_state_dict = build_distributed_state_dict_from_consolidated(
            model, state_dict, model_parallel_world_size=MODEL_PARALLEL_SIZE,
        )
        # Wrap model with PT-native APIs + load
        model = FSDP(model)
        FSDP.set_state_dict_type(StateDictType.SHARDED_STATE_DICT)
        model.load_state_dict(sharded_state_dict)
        ```

    Note: Please make sure to pass an unsharded model as the model arg! Otherwise, things will not
    work.

    This distributed state dict is a mapping of FQN: ShardedTensor/DTensor. It will be replaced with
    DTensor once DTensor 2D checkpoint format is fully rolled out.

    Note: This has only been tested for loading state_dict into PT-D FSDP sharded_state_dict for now.
    """
    torch._C._log_api_usage_once("build_distributed_state_dict")
    dist_state_dict = {}
    ref_state_dict = model.state_dict()
    grp_gloo = dist.new_group(backend="gloo")
    # TODO: this should be the FSDP device mesh
    mesh = (
        DeviceMesh(
            device_type="cuda",
            mesh=list(range(dist.get_world_size())),
        )
        if use_dtensor
        else None
    )
    input_dtypes = {v.dtype for v in consolidated_state_dict.values()}
    logging.warning(f"input_dtypes {input_dtypes}")
    model_parallel_group, _ = dist.new_subgroups(group_size=model_parallel_world_size)
    for fqn, tensor in consolidated_state_dict.items():
        # Hack for buffer
        if "rope.freqs" in fqn:
            dist_state_dict[fqn] = tensor.clone()
            continue
        if _is_tp_sharded(fqn):
        
            tensor, _ = _unshard_param(
                ref_state_dict,
                fqn,
                model_parallel_group,
                grp_gloo,
                tensor,
                tensor.shape,
            )
        if use_dtensor:
       
            assert mesh is not None
            tensor = _ext_chunk_dtensor(
                tensor=tensor.contiguous(),
                rank=dist.get_rank(),
                device_mesh=mesh,
            )
         
        else:
         
            tensor = _ext_chunk_tensor(
                tensor=tensor.contiguous(),
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                num_devices_per_node=torch.cuda.device_count(),  # TODO: this is not accurate if user set CUDA_VISIBLE_DEVICES
                pg=dist.distributed_c10d._get_default_group(),  # TODO: this should be the FSDP process group
            )
        # try:
        #     if isinstance(tensor, DTensor):
        #         print(f"{fqn} is DTensor")
        # except:
        #     print(f"{fqn} is not DTensor")
        dist_state_dict[fqn] = tensor
    # assert isinstance(tensor, DTensor), f"The tensor at fqn '{fqn}' is not a DTensor."
    dtypes = {v.dtype for v in dist_state_dict.values()}
    logging.warning(f"Made dist_state_dict with dtypes {dtypes}")
    return dist_state_dict

