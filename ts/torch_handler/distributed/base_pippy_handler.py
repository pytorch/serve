"""
Base default handler to load large models using PyTorch Native PiPPy.
"""
import os
from abc import ABC

from ts.handler_utils.distributed.pt_pippy import initialize_rpc_workers
from ts.torch_handler.base_handler import BaseHandler


class BasePippyHandler(BaseHandler, ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        initialize_rpc_workers(self.local_rank, self.world_size)
