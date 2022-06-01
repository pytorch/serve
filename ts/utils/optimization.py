from abc import abstractmethod
import torch 

OPTIMIZATIONS = {}

def optimization_registry(cls):  
    assert cls.__name__.endswith('Optimization'), "The name of subclass of Optimization should end with \'Optimization\' substring."
    if cls.__name__[:-len('Optimization')].lower() in OPTIMIZATIONS:
        raise ValueError('Cannot have two optimizations with the same name')
    OPTIMIZATIONS[cls.__name__[:-len('Optimization')].lower()] = cls
    return cls

class Optimization(object):
    def __init__(self, model):
        self.model = model
        
    @abstractmethod
    def optimize(self, model : torch.nn.Module, **kwargs) -> torch.nn.Module:
        raise NotImplementedError("This is an abstract base class, you need to call or create your own runtime")
