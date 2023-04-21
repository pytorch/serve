"""
Base default handler to load large models using PyTorch Native PiPPy.
"""
import os
from abc import ABC

import torch

from ts.handler_utils.distributed.pt_pippy import initialize_rpc_workers
from ts.torch_handler.base_handler import BaseHandler


class BasePippyHandler(BaseHandler, ABC):
    """
    Base default handler to set up rpc workers for PiPPy large model inference
    """

    def initialize(self, ctx):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        n_devs = torch.cuda.device_count()
        self.device = self.local_rank % n_devs
        initialize_rpc_workers(self.local_rank, self.world_size, ctx)
