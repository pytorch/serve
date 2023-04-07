import os
from abc import ABC

from ts.context import Context
from ts.handler_utils.distributed.deepspeed import get_ds_engine
from ts.torch_handler.base_handler import BaseHandler, logger


class BaseDeepSpeedHandler(BaseHandler, ABC):
    """
    Base default DeepSpeed handler.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx: Context):
        ds_engine = get_ds_engine(self.model, ctx)
        self.model = ds_engine.module
        self.device = int(os.getenv("LOCAL_RANK", 0))
        self.initialized = True
        logger.info("Model %s loaded successfully", ctx.model_name)
