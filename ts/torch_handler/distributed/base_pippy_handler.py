"""
Base default handler to load large models using PyTorch Native PiPPy.
"""
from abc import ABC
import importlib.util
import logging
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
from ts.handler_utils.distributed.pt_pippy import get_pipline_driver, initialize_rpc_workers

class BasePippyHandler(BaseHandler,ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        initialize_rpc_workers( self.local_rank,self.world_size)


# if __name__=="__main__":
#     handler = BasePippyHandler()
#     print(dir(handler))
