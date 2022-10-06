from pathlib import Path
import copy
from .config import configuration_registry, Conf
import subprocess
import yaml 
import torch 
from dataclasses import dataclass, asdict

DTYPES = ['float32', 'bfloat16']

DEFAULT_CFG = {'dtype': 'float32', 
              'channels_last': True
              }

@dataclass
@configuration_registry
class IPEXConf(Conf):
    """The Intel® Extension for PyTorch* (IPEX) optimization configuration.
    
    Attributes:
        dtype (str): # optional. supported values are float32, bfloat16, int8. default value is float32.
        channels_last (bool): # optional. supported values are True, False. default value is True.
    """
    dtype: str = 'float32'
    channels_last: bool = True
        
    def __post_init__(self):
        super().__init__(self.cfg_file_path)
        assert Path(self.cfg_file_path).exists(), "{} does not exist".format(self.cfg_file_path)
        cfg = self.read_conf(self.cfg_file_path)
        cfg = self._convert_cfg(cfg, copy.copy(DEFAULT_CFG))
        self.dtype = cfg['dtype']
        self.dtype = self.dtype.lower()
        assert self.dtype in DTYPES, "dtype {} is NOT supported".format(self.dtype)
        
        self.channels_last = cfg['channels_last']
        assert isinstance(self.channels_last, bool), "channels last must be type bool"
        
    def _convert_cfg(self, src, dst):
        """Helper function to merge user defined dict into default dict.
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
            if key in dst:
                if dst[key] != src[key]:
                    dst[key] = src[key]
        return dst