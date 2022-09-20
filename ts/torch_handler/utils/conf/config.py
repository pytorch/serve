from abc import abstractmethod
from pathlib import Path
from schema import Schema
import copy
import yaml
from dataclasses import dataclass

"""CONFIGURATIONS variable is a dictionary used to store all implemented Conf subclasses.
   See __init__.py for details.
"""
CONFIGURATIONS = {}

def configuration_registry(cls):
    """The class decorator used to register all Conf subclasses.
    
    Args:
        cls (class): The class of register.
    Returns:
        cls: The class of register.
    """
    assert cls.__name__.endswith('Conf'), "The name of subclass of Conf should end with \'Conf\' substring."
    if cls.__name__[:-len('Conf')].lower() in CONFIGURATIONS:
        raise ValueError('Cannot have two configurations with the same name')
    CONFIGURATIONS[cls.__name__[:-len('Conf')].lower()] = cls
    return cls

@dataclass
class Conf:
    """The base class of optimization configuration.
    
    Attributes:
        cfg_file_path (str): The path to the yaml configuration file.    
    """
    cfg_file_path: str
    
    def read_conf(self, cfg_file_path):
        """Load a configiguration file following yaml syntax.
           Args:
               cfg_file_path (string): The path to the yaml configuration file.
        """
        with open(cfg_file_path, 'r') as f:
            content = f.read()
            conf = yaml.safe_load(content)
            if conf is None: conf = dict()
            return conf