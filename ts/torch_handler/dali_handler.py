from abc import ABC

from ts.context import Context
from ts.handler_utils.dali import get_dali_pipeline
from ts.torch_handler.base_handler import BaseHandler


class DaliHandler(BaseHandler, ABC):
    """
    Base default DeepSpeed handler.
    """

    def initialize(self, ctx: Context):
        self.pipeline = get_dali_pipeline()
