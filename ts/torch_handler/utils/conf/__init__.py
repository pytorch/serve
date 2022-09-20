from .config import CONFIGURATIONS
from os.path import dirname, basename, isfile, join
import glob

"""All optimization configurations are located in this directory, ts/torch_handler/conf/.
   
   Currently, optimization configurations supported by TorchServe include Intel® Extension for PyTorch* (IPEX).
   
   To add a new optimization configuration, implement a new Conf subclass under this directoy in a new Python file.
   Decorate the new Conf subclass with @configuration_registry.
   The naming convention of the new Conf subclass should end in 'Conf', something like 
   
   @configuration_registry
       class CustomConf(Conf):
   
   The naming convention of the new Python file where the new Conf subclass is implemented should end in 'config.py', something like 'custom_config.py'.
   
   CONFIGURATIONS variable is a dictionary used to store all implemented Conf subclasses. 
   CONFIGURATIONS can be imported (from ts.torch_handler.utils.conf import CONFIGURATIONS) in the base_handler or in a customized handler.
   The new optimization configuration can be accessed from CONFIGURATIONS by the name of the new Conf subclass, something like CONFIGURATIONS['custom'].
"""

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and f.endswith('config.py'):
        __import__(basename(f)[:-3], globals(), locals(), level=1)

__all__ = ["CONFIGURATIONS"]