from pathlib import Path
import copy
from .config import configuration_registry, Conf
import subprocess
import yaml 
import torch 
from dataclasses import dataclass, asdict
import typing as t

DTYPES = ['float32', 'bfloat16', 'int8']
QUANTIZATION_APPROACHES = ['static', 'dynamic']
TORCHSCRIPT_APPROACHES = ['trace']

DEFAULT_CFG = {'dtype': 'float32', 
              'channels_last': True, 
              'quantization': {'approach': None, 'calibration_dataset': None}, 
              'torchscript': {'approach': None, 'example_inputs': None}
              }

def _load_file_path(f):
    assert Path(f).exists(), "{} does not exist".format(f)
    data = torch.load(f)
    return data

@dataclass
class QuantizationConf:
    approach: str = None
    calibration_dataset: str = None
    
    def __post_init__(self):
        if self.approach is not None:
            self.approach = self.approach.lower()
            assert self.approach in QUANTIZATION_APPROACHES, "quantization approach {} is NOT supported".format(self.approach)
        
        if self.approach == 'static':
            assert self.calibration_dataset is not None, "path to calibration_dataset must be provided for static quantization"
        
        if self.calibration_dataset is not None:
            self.calibration_dataset = _load_file_path(self.calibration_dataset)
            assert all(isinstance(x, tuple) for x in self.calibration_dataset) or all(isinstance(x, torch.Tensor) for x in self.calibration_dataset), "calibration_dataset must be a list of tuple(s) or a list of torch.Tensor(s)"

@dataclass 
class TorchscriptConf:
    approach: str = None
    example_inputs: str = None
    
    def __post_init__(self):
        if self.approach is not None:
            self.approach = self.approach.lower()
            assert self.approach in TORCHSCRIPT_APPROACHES, "torchscript approach {} is NOT supported".format(self.approach)
        
        if self.approach == 'trace':
            assert self.example_inputs is not None, "path to example_inputs must be provided for TorchScript trace"
        
        if self.example_inputs is not None:
            self.example_inputs = _load_file_path(self.example_inputs)
            assert isinstance(self.example_inputs, tuple) or isinstance(self.example_inputs, torch.Tensor), "example_inputs must be of type tuple or torch.Tensor"

@dataclass
@configuration_registry
class IPEXConf(Conf):
    """The Intel® Extension for PyTorch* (IPEX) optimization configuration.
    
    Attributes:
        dtype (str): # optional. supported values are float32, bfloat16, int8. default value is float32.
        channels_last (bool): # optional. supported values are True, False. default value is True.
        quantization:
          approach (str): # mandatory if int8 dtype, otherwise not applicable. supported values are static, dynamic. default value is None.
          calibration_dataset (str): # mandatory if static approach. path to your calibration dataset if static quantization. default value is None. 
        torchscript:
          approach (str): # optional. supported values is trace. default value is None. 
          example_inputs (str): # mandatory if trace approach. path to your example_inputs if TorchScript trace. default value is None. 
    """
    dtype: str = 'float32'
    channels_last: bool = True
    quantization: QuantizationConf = QuantizationConf()
    torchscript: TorchscriptConf = TorchscriptConf()
        
    def __post_init__(self):
        super().__init__(self.cfg_file_path)
        assert Path(self.cfg_file_path).exists(), "{} does not exist".format(self.cfg_file_path)
        cfg = self.read_conf(self.cfg_file_path)
        cfg = self._convert_cfg(cfg, DEFAULT_CFG)
        
        self.dtype = cfg['dtype'] 
        
        self.dtype = self.dtype.lower()
        assert self.dtype in DTYPES, "dtype {} is NOT supported".format(self.dtype)
        assert bool(self.dtype == 'int8') == bool(cfg['quantization']['approach'] is not None), "quantization approach must be provided for INT8 dtype, and quantization is supported for INT8 dtype only"
        
        self.channels_last = cfg['channels_last']
        
        assert isinstance(self.channels_last, bool), "channels last must be type bool"

        self.quantization = QuantizationConf(cfg['quantization']['approach'], cfg['quantization']['calibration_dataset'])
        self.torchscript = TorchscriptConf(cfg['torchscript']['approach'], cfg['torchscript']['example_inputs'])
        
    def _convert_cfg(self, src, dst):
        """Helper function to recursively merge user defined dict into default dict.
           If the key in src doesn't exist in dst, then add this key and value
           pair to dst.
           Otherwise, if the key in src exists in dst, then override the value in dst with the
           value in src.
        Args:
            src (dict): The source dict merged from
            dst (dict): The source dict merged to
        Returns:
            dict: The merged dict from src to dst
        """
        for key in src:
            if isinstance(src[key], dict):
                self._convert_cfg(src[key], dst[key])
            else:
                dst[key] = src[key]
        return dst