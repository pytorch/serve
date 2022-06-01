from .optimization import optimization_registry, Optimization
import os
import torch
import logging
import subprocess 

logger = logging.getLogger(__name__)

@optimization_registry
class IPEXOptimization(Optimization):
    def __init__(self, model, ipex_enabled, onednn_graph_fusion_enabled):
        super().__init__(model)
        
        self.ipex_enabled = ipex_enabled
        self.onednn_graph_fusion_enabled = onednn_graph_fusion_enabled
        
        self.dtype = os.environ.get("TS_IPEX_DTYPE", "float32")
        self.channel_last = os.environ.get("TS_IPEX_CHANNEL_LAST", "true")
        
        self.torchscript = os.environ.get("TS_IPEX_TORCHSCRIPT", "true")
        self.input_tensor_shapes = os.environ.get("TS_IPEX_INPUT_TENSOR_SHAPES", "null")
        self.input_tensor_dtype = os.environ.get("TS_IPEX_INPUT_TENSOR_DTYPE", "null")
        self.qscheme = os.environ.get("TS_IPEX_QSCHEME", "per_tensor_affine")
        
        if self.dtype == "bfloat16" and not self.is_bf16_supported():
            self.dtype = "float32"
            logger.warn("You have specified bfloat16 dtype, but bfloat16 dot-product hardware accelerator is not supported in your current hardware. Proceeding with float32 dtype instead.")
            
        if self.dtype == "int8" and self.torchscript == "false":
            logger.error("Quantization in IPEX takes advantage of oneDNN graph API. This requires to be executed with TorchScript mode. Please set ipex_torchscript to true.")
            exit(-1)
        if self.onednn_graph_fusion_enabled and self.torchscript == "false":
            logger.error("oneDNN graph fusion requires torchscript mode. Please set ipex_torchscript to true.")
            exit(-1)
        if self.torchscript == "true" and self.input_tensor_shapes == "null" or self.torchscript == "true" and self.input_tensor_dtype == "null":
            logger.error("Please specify both ipex_input_tensor_shapes and ipex_input_tensor_dtype to do calibration for torchscript mode")
            exit(-1)

        self.TORCH_DTYPE = {"TYPE_FP16": torch.float16,
                            "TYPE_FP32": torch.float32,
                            "TYPE_FP64": torch.float64,
                        
                            "TYPE_BF16": torch.bfloat16,
                        
                            "TYPE_UINT8": torch.uint8,
                        
                            "TYPE_INT8": 	torch.int8,
                            "TYPE_INT16": torch.int16,
                            "TYPE_INT32": torch.int32,
                            "TYPE_INT64": torch.int64
                            }
                    
        self.TORCH_QSCHEME = {"per_tensor_affine": torch.per_tensor_affine,
                              "per_tensor_symmetric": torch.per_tensor_symmetric
                              }
    
    def is_bf16_supported(self):
        proc1 = subprocess.Popen(['lscpu'], stdout=subprocess.PIPE)
        proc2 = subprocess.Popen(['grep', 'Flags'], stdin=proc1.stdout, stdout=subprocess.PIPE)
        proc1.stdout.close()
        out = proc2.communicate()
        return 'bf16' in str(out)
        
    def get_dummy_tensors(self):
        x = []
        for input_tensor_shape in self.input_tensor_shapes.split(";"):
            input_tensor_shape = tuple(int(_) for _ in input_tensor_shape.split(","))
            dummy_tensor = torch.ones(input_tensor_shape, dtype=self.TORCH_DTYPE[self.input_tensor_dtype])
            if self.channel_last == "true":
                dummy_tensor = dummy_tensor.to(memory_format=torch.channels_last)
            x.append(dummy_tensor)
        return x 

    def warmup_forward(self, m, *x, profiling_count=2):
        for _ in range(profiling_count):
            m(*x)
        return 
    
    def get_torchscripted_model(self, m, x, freeze=True):
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad():
            if not isinstance(m, torch.jit.ScriptModule):
                try:
                    traced = torch.jit.trace(m, x)
                except RuntimeError:
                    try: 
                        traced = torch.jit.trace(m, x, check_trace=False, strict=False)
                    except:
                        logger.error("Tracing the model failed. Pleace make sure the model is traceable")
                        exit(-1)
            else:
                traced = m
            
            if not isinstance(traced, torch.jit.RecursiveScriptModule):
                freezed = torch.jit.freeze(traced)
            else:
                freezed = traced 
            self.warmup_forward(freezed, *x)
            
        return freezed 
        
    def optimize(self):
        if self.onednn_graph_fusion_enabled:
            torch.jit.enable_onednn_fusion(True)
        
        # channel last 
        if self.channel_last == "true":
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # ipex optimization 
        if self.ipex_enabled:    
            import intel_extension_for_pytorch as ipex

            if self.dtype == "float32":
                self.model = ipex.optimize(self.model, dtype=torch.float32)
            elif self.dtype == "bfloat16":
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
            else: # int8
                from intel_extension_for_pytorch.quantization import prepare, convert
                from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
                
                x = self.get_dummy_tensors()
                
                qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=self.TORCH_QSCHEME[self.qscheme], dtype=torch.quint8),weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
                self.model = prepare(self.model, qconfig, example_inputs=x, inplace=False)
                # calibration 
                n_iter = 100
                for i in range(n_iter):
                    self.model(*x)
                # conversion 
                self.model = convert(self.model)
    
        # torchscript
        if self.torchscript == "true":
            x = self.get_dummy_tensors()
            
            if self.dtype == "float32" or self.dtype == "int8":
                self.model = self.get_torchscripted_model(self.model, x)
            else: # bfloat16
                with torch.cpu.amp.autocast():
                    self.model = self.get_torchscripted_model(self.model, x)
        
        return self.model