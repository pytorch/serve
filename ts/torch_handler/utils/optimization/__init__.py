from .optimization import OPTIMIZATIONS
from os.path import dirname, basename, isfile, join
import glob

"""All optimizations are located in this directory, ts/torch_handler/optimization/.
   
   Currently, optimizations supported by TorchServe include Intel® Extension for PyTorch* (IPEX).
   
   To add a new optimization, implement a new Optimization subclass under this directoy in a new Python file.
   Decorate the new Optimization subclass with @optimization_registry.
   The naming convention of the new Optimization subclass should end in 'Optimization', something like 
   
   @optimization_registry
       class CustomOptimization(Conf):
   
   The naming convention of the new Python file where the new Optimization subclass is implemented should end in 'optimization.py', something like 'custom_optimization.py'.
   
   OPTIMIZATIONS variable is a dictionary used to store all implemented Optimization subclasses. 
   OPTIMIZATIONS can be imported (from ts.torch_handler.utils.conf import OPTIMIZATIONS) in the base_handler or in a customized handler.
   The new optimization can be accessed from OPTIMIZATIONS by the name of the new Optimization subclass, something like OPTIMIZATIONS['custom'].
"""

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and f.endswith('optimization.py'):
        __import__(basename(f)[:-3], globals(), locals(), level=1)

__all__ = ["OPTIMIZATIONS"]