from .optimization import optimization_registry, Optimization
from ..conf.ipex_config import Conf
import os
import torch
import logging
import subprocess 

logger = logging.getLogger(__name__)

@optimization_registry
class IPEXOptimization(Optimization):
    """The Intel® Extension for PyTorch* (IPEX) Optimization.
    
    Args:
        cfg (Conf): the optimization configuration.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.dtype = cfg.dtype
        self.channels_last = cfg.channels_last

    def optimize(self, model):
        """Apply Intel® Extension for PyTorch* (IPEX) optimizations to the given model (nn.Module).
           
        Args:
            model (torch.nn.Module): The model to optimize. 
        Returns:
            torch.nn.Module: The optimized model.
        """
        import intel_extension_for_pytorch as ipex
        
        # channel last 
        if self.channels_last:
            logger.info("converting to channels last memory format")
            model = model.to(memory_format=torch.channels_last)
        
        # dtype 
        if self.dtype == 'float32':
            logger.info("optimizing model with data type torch.float32")
            model = ipex.optimize(model, dtype=torch.float32)
        else: # bfloat16
            logger.info("optimizing model with data type torch.bfloat16")
            model = ipex.optimize(model, dtype=torch.bfloat16)

        return model