from schema import Schema, And, Use, Optional, Or, Hook
from pathlib import Path
import copy
from .config import configuration_registry, Conf
import subprocess
import torch 

DTYPES = ['float32', 'bfloat16', 'int8']
QUANTIZATION_APPROACHES = ['static', 'dynamic']
TORCHSCRIPT_APPROACHES = ['trace', 'script']

def _valid_quantization_approach(data):
    data = data.lower()
    assert data in QUANTIZATION_APPROACHES, "quantization approach {} is NOT supported".format(data)
    return data 

def _load_file_path(data):
    if isinstance(data, str):
        assert Path(data).exists(), "{} does not exist".format(data)
        data = torch.load(data)
    return data
    
quantization_schema = Schema({
                            Optional('approach', default='static'): And(str, Use(_valid_quantization_approach)),
                            
                            'calibration_dataset': And(Or(str, And(list, lambda s: all(isinstance(i, tuple) for i in s)), And(list, lambda s: all(isinstance(i, torch.Tensor) for i in s))), Use(_load_file_path))
                            })

def _valid_torchscript_approach(data):
    data = data.lower()
    assert data in TORCHSCRIPT_APPROACHES, "torchscript approach {} is NOT supported".format(data)
    return data 

def _valid_torchscript_schema(key, scope, error):
    if scope[key]['approach'] == 'trace':
        assert 'example_inputs' in scope[key], "make sure to provide example_inputs with TorchScript trace"

torchscript_schema = Schema({
                            Optional('approach', default='trace'): And(str, Use(_valid_torchscript_approach)),
                            
                            Optional('example_inputs'): And(Or(str, tuple, torch.Tensor), Use(_load_file_path))
                            })

def _valid_dtype(data):
    data = data.lower()
    assert data in DTYPES, "dtype {} is NOT supported".format(data)
    if data == 'bfloat16': 
        assert is_bf16_supported(), "You've selected bfloat16 dtype, but bfloat16 dot-product hardware accelerator is not supported in your current hardware. Please select float32 or int8 dtype, or switch to bfloat16 supported hardware."
    return data 

def is_bf16_supported():
    proc1 = subprocess.Popen(['lscpu'], stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(['grep', 'Flags'], stdin=proc1.stdout, stdout=subprocess.PIPE)
    proc1.stdout.close()
    out = proc2.communicate()
    return 'bf16' in str(out)
        
def _valid_int8_schema(key, scope, error):
    if scope[key] == 'int8':
        assert 'quantization' in scope, "quantization schema must be provided for int8 dtype"
        assert 'calibration_dataset' in scope['quantization'], "calibration_dataset must be provided for int8 dtype"
              
schema = Schema({
                Hook('dtype', handler=_valid_int8_schema): object,
                Optional('dtype', default='float32'): And(str, Use(_valid_dtype)),
                
                Optional('channels_last', default=True): And(bool, lambda s: s in [True, False]),

                Optional('quantization'): quantization_schema, 
                
                Hook('torchscript', handler=_valid_torchscript_schema): object,
                Optional('torchscript'): torchscript_schema
                })

@configuration_registry
class IPEXConf(Conf):
    def __init__(self, cfg_file_path):
        super().__init__(cfg_file_path)
    
    def get_usr_cfg(self, cfg_file_path):
        usr_cfg = schema.validate(self._convert_cfg(self._read_conf(cfg_file_path, schema), copy.deepcopy(schema.validate(dict()))))
        return usr_cfg 