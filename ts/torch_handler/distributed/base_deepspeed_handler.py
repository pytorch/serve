import os
from abc import ABC

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler


class BaseDeepSpeedHandler(BaseHandler, ABC):
    """
    Base default DeepSpeed handler.
    """

    def initialize(self, ctx: Context):
        self.device = int(os.getenv("LOCAL_RANK", 0))
