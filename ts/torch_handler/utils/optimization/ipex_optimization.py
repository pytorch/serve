from .optimization import optimization_registry, Optimization
from ..conf.ipex_config import Conf
import os
import torch
import logging
import subprocess 

logger = logging.getLogger(__name__)

@optimization_registry
class IPEXOptimization(Optimization):
    def __init__(self, model, cfg):
        super().__init__(model)
        
        self.model = model
        self.dtype = cfg['dtype'] 
        self.channels_last = cfg['channels_last']
        
        self.quantization = False 
        if 'quantization' in cfg:
            self.quantization = True 
            self.quantization_approach = cfg['quantization']['approach']
            self.quantization_calibration_dataset = cfg['quantization']['calibration_dataset']
        
        self.torchscript = False 
        if 'torchscript' in cfg:
            self.torchscript = True 
            self.torchscript_approach = cfg['torchscript']['approach'] 
            self.torchscript_example_inputs = cfg['torchscript']['example_inputs']

    def optimize(self):
        import intel_extension_for_pytorch as ipex
        
        # channel last 
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # dtype 
        if self.dtype == 'float32':
            self.model = ipex.optimize(self.model, dtype=torch.float32)
        elif self.dtype == 'bfloat16':
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
        else: # int8
            from intel_extension_for_pytorch.quantization import prepare, convert
            
            if self.quantization_approach == 'static':
                qconfig = ipex.quantization.default_static_qconfig
                
                # prepare and calibrate 
                self.model = prepare(self.model, qconfig, example_inputs=self.quantization_calibration_dataset[0], inplace=False)
                for x in self.quantization_calibration_dataset:
                    if isinstance(x, torch.Tensor):
                        x = (x,)
                    self.model(*x)
                
                # convert 
                self.model = convert(self.model)
                
            else: # dynamic 
                qconfig = ipex.quantization.default_dynamic_qconfig
                
                # prepare
                self.model = prepare(self.model, qconfig, example_inputs=self.quantization_calibration_dataset[0])
            
                # convert and deploy 
                self.model = convert(self.model)
        
        # torchscript 
        if self.torchscript:
            with torch.no_grad():
                if self.torchscript_approach == 'trace':
                    try:
                        self.model = torch.jit.trace(self.model, example_inputs=self.torchscript_example_inputs)
                    except:
                        try: 
                            self.model = torch.jit.trace(self.model, example_inputs=self.torchscript_example_inputs, check_trace=False, strict=False)
                        except:
                            logger.error("TorchScript tracing the model failed. Make sure the model is traceable.")
                            exit(-1)
                else: # script 
                    try:
                        self.model = torch.jit.script(self.model)
                    except:
                        logger.error("TorchScript scripting the model failed. Make sure the model is scriptable.")
                        exit(-1)
                    
                self.model = torch.jit.freeze(self.model)
        
        return self.model