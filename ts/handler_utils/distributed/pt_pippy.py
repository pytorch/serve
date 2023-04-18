import importlib
import inspect
import logging
import os

import pippy
import torch
import torch.distributed.rpc as rpc

pippy_installed = importlib.util.find_spec("pippy") is not None

if pippy_installed:
    from pippy import split_into_equal_size
    from pippy.hf import PiPPyHFTracer, inject_pipeline_forward


logger = logging.getLogger(__name__)


def initialize_rpc_workers(local_rank, world_size, ctx):
    # Get RPC configuration options from model YAML config
    rpc_timeout = ctx.model_yaml_config["pippy"]["rpc_timeout"]
    num_worker_threads = ctx.model_yaml_config["pippy"]["num_worker_threads"]
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=num_worker_threads, rpc_timeout=rpc_timeout
    )

    # Set up device mapping for RPC workers
    n_devs = torch.cuda.device_count()
    dev_id = local_rank % n_devs
    for i in range(world_size):
        options.set_device_map(f"worker{i}", {dev_id: i % n_devs})

    # Initialize RPC worker
    logger.info(f"rank = {local_rank} pid/device = " f"{os.getpid()}/{dev_id}")
    rpc.init_rpc(
        f"worker{local_rank}",
        rank=local_rank,
        world_size=world_size,
        rpc_backend_options=options,
    )


def get_pipeline_driver(model, world_size, ctx):
    """Returns a pipeline driver for the given model.
    Args:
        model (torch.nn.Module): The model to pipeline.
        world_size (int): The number of pipeline stages.
        ctx (Context): The context containing configuration information.
    Returns:
        torch.nn.Sequential: The pipeline driver for the model.
    """
    # Extract configuration parameters from the context
    chunks = ctx.model_yaml_config["pippy"]["chunks"]
    input_names = ctx.model_yaml_config["pippy"]["input_names"]
    model_type = ctx.model_yaml_config["pippy"]["model_type"]

    # Set the model to evaluation mode
    model.eval()

    # Extract the concrete arguments for the model's forward method
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    logger.info("Initializing the model pipeline")

    # Create a tracer if necessary
    tracer = PiPPyHFTracer() if model_type == "HF" else None

    # Add deprecated_arguments to concrete_args if necessary
    if model_type == "HF" and "bloom" in str(model.__class__):
        concrete_args.setdefault("deprecated_arguments", {})

    # Compile the pipeline using PiPPy
    split_policy = split_into_equal_size(world_size)
    pipe_driver, stage_mode = pippy.all_compile(
        model,
        num_ranks=world_size,
        num_chunks=chunks,
        schedule="FillDrain",
        split_policy=split_policy,
        tracer=tracer,
        concrete_args=concrete_args,
    )

    # Inject the pipeline forward method if necessary
    if model_type == "HF":
        inject_pipeline_forward(model, pipe_driver)
        return model
    else:
        return pipe_driver
