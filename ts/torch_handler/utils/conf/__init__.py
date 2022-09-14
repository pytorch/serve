from .config import CONFIGURATIONS
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and f.endswith('config.py'):
        __import__(basename(f)[:-3], globals(), locals(), level=1)

__all__ = ["CONFIGURATIONS"]