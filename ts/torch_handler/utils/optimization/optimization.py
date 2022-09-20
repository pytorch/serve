from abc import abstractmethod
import torch 

"""OPTIMIZATIONS variable is a dictionary used to store all implemented Optimization subclasses.
   See __init__.py for details.
"""
OPTIMIZATIONS = {}

def optimization_registry(cls): 
    """The class decorator used to register all Conf subclasses.
    
    Args:
        cls (class): The class of register.
    Returns:
        cls: The class of register.
    """ 
    assert cls.__name__.endswith('Optimization'), "The name of subclass of Optimization should end with \'Optimization\' substring."
    if cls.__name__[:-len('Optimization')].lower() in OPTIMIZATIONS:
        raise ValueError('Cannot have two optimizations with the same name')
    OPTIMIZATIONS[cls.__name__[:-len('Optimization')].lower()] = cls
    return cls

class Optimization(object):
    """The base class of optimization.
    
    Attributes:
        cfg (Conf): The optimization configuration.    
    """
    def __init__(self, cfg):
        self.cfg = cfg
        
    @abstractmethod
    def optimize(self, model : torch.nn.Module, **kwargs) -> torch.nn.Module:
        """The optimization function. 
        This is where all custom optimizations following their configuration are implemented.
           
        Args:
            model (torch.nn.Module): The model to optimize.
            
        Returns:
            torch.nn.Module: The optimized model.
        """
        raise NotImplementedError("This is an abstract base class, you need to call or create your own.")