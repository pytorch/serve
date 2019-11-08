
"""
Model services code
"""
import warnings

from . import model_service
from . import mxnet_model_service
from . import mxnet_vision_service

warnings.warn("Module mms.model_service is deprecated, please migrate to model archive 1.0 format.",
              DeprecationWarning, stacklevel=2)
