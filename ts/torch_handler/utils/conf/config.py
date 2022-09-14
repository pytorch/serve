from abc import abstractmethod
from pathlib import Path
from schema import Schema
import copy
import yaml

CONFIGURATIONS = {}

def configuration_registry(cls):  
    assert cls.__name__.endswith('Conf'), "The name of subclass of Conf should end with \'Conf\' substring."
    if cls.__name__[:-len('Conf')].lower() in CONFIGURATIONS:
        raise ValueError('Cannot have two configurations with the same name')
    CONFIGURATIONS[cls.__name__[:-len('Conf')].lower()] = cls
    return cls

class Conf(object):
    """config parser.
    Args:
        cfg_file_path (string): The path to the yaml configuration file.
    """
    def __init__(self, cfg_file_path):
        assert Path(cfg_file_path).exists(), "{} does not exist".format(cfg_file_path)
    
    @abstractmethod
    def get_usr_cfg(self, cfg_file_path : str, **kwargs) -> dict:
        """Returns user's validated confiugration file in dict.
           Args:
               cfg_file_path (string): The path to the yaml configuration file.
        """
        raise NotImplementedError("This is an abstract base class, you need to call or create your own.")

    def _read_conf(self, cfg_file_path, schema):
        """Load a configiguration file following yaml syntax.
           Args:
               cfg_file_path (string): The path to the yaml configuration file.
               schema (Schema): The schema to validate the yaml configuration file 
        """
        try:
            with open(cfg_file_path, 'r') as f:
                content = f.read()
                conf = yaml.safe_load(content)
                if conf is None: conf = dict()
                validated_conf = schema.validate(conf)
            return validated_conf

        except:
            raise RuntimeError("The yaml configuration file format is not correct. Refer to the document.")

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
            dst[key] = src[key]
        return dst 