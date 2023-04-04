import inspect
import logging
import os

import pippy
import torch
import torch.distributed.rpc as rpc
from pippy import split_into_equal_size
from pippy.hf import PiPPyHFTracer

logger = logging.getLogger(__name__)


def initialize_rpc_workers(local_rank, world_size):
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=512, rpc_timeout=1800)
    n_devs = torch.cuda.device_count()
    dev_id = local_rank % n_devs
    for i in range(world_size):
        options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
    print(f"rank = {local_rank} pid/device = " f"{os.getpid()}/{dev_id}")
    rpc.init_rpc(
        f"worker{local_rank}",
        rank=local_rank,
        world_size=world_size,
        rpc_backend_options=options,
    )


def get_pipline_driver(model, world_size, input_names, model_type, chunks):
    model.eval()
    split_policy = split_into_equal_size(world_size)
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    logger.info("initializing the model pipline")
    if model_type == "HF":
        tracer = PiPPyHFTracer()
    else:
        tracer = None
    pipe_driver, stage_mode = pippy.all_compile(
        model,
        num_ranks=world_size,
        num_chunks=chunks,
        schedule="FillDrain",
        split_policy=split_policy,
        tracer=tracer,
        concrete_args=concrete_args,
    )
    return pipe_driver
