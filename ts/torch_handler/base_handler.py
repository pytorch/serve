"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""

import abc
import logging
import os
import types
from typing import Any

import torch
from pkg_resources import packaging

from ts.handler_utils import (
    BaseHandle,
    BaseInference,
    BaseInit,
    BasePostprocess,
    BasePreproc,
)

if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.8.1"):

    PROFILER_AVAILABLE = True
else:
    PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


ipex_enabled = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    try:

        ipex_enabled = True
    except ImportError as error:
        logger.warning(
            "IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX."
        )


HANDLER_KW = "initialize", "preprocess", "inference", "postprocess", "handle"


class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        self.initialized = False
        super().__setattr__("__assigned__", set())
        self.initialize = BaseInit()
        self.preprocess = BasePreproc()
        self.inference = BaseInference()
        self.postprocess = BasePostprocess()
        self.handle = BaseHandle()
        self.model = None
        self.mapping = None
        self.device = None
        self.context = None
        self.model_pt_path = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.profiler_args = {}

    def __getattribute__(self, name: str):
        assert "initialized" in object.__getattribute__(
            self, "__dict__"
        ), "Can not use handler before calling BaseHandler.__init__()."
        return object.__getattribute__(self, name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in HANDLER_KW:
            assert "__assigned__" in super().__getattribute__(
                "__dict__"
            ), "Can not use handler before calling BaseHandler.__init__()."
            # Ignore assignment if method is present due to inheritance
            if hasattr(self, __name) and __name not in self.__assigned__:
                return None
            return self._bind_method(__name, __value)
        return super().__setattr__(__name, __value)

    def _bind_method(self, name, method):
        self.__assigned__.add(name)
        self.__dict__[name] = types.MethodType(method, self)
