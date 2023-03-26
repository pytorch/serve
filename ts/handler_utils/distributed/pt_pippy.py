import importlib.util

import os
import time
import torch
from pkg_resources import packaging
from ts.torch_handler.base_handler import BaseHandler
import torch.distributed.rpc as rpc
import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import TensorChunkSpec
from pippy import split_on_size_threshold, split_into_equal_size
import inspect

def initialize_rpc_workers(local_rank,world_size):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=512,
        rpc_timeout=1800
    )
    n_devs = torch.cuda.device_count()
    dev_id = local_rank % n_devs 
    for i in range (world_size):
        options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
    print(
        f"rank = {local_rank} pid/device = "
        f"{os.getpid()}/{dev_id}"
    )
    rpc.init_rpc(f"worker{local_rank}",
                    rank=local_rank,
                    world_size=world_size,
                    rpc_backend_options=options)

def load_hf_causuallm_models_from_checkpoints(model_dir):
    with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")
    model = AutoModelForCausalLM.from_pretrained(
            model_dir + "/model", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(
            model_dir + "/model", return_tensors="pt"
        )
    return model, tokenizer
def load_hf_causuallm_models_from_pretrained(model_name):
    model = AutoModelForCausalLM.from_pretrained(
            model_name, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(
            model_name, return_tensors="pt"
        )
    return model, tokenizer

def get_pipline_driver(model,world_size, input_names, model_type, chunks):
    model.eval()
    split_policy = split_into_equal_size(world_size)
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    print('Instantiating model Pipeline')
    model_init_start = time.time()
    if model_type=="HF":
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
    model_init_end = time.time()
    logger.info("Model init time took {} ms".format(round((stop_time - start_time) * 1000, 2)))
    return pipe_driver



