"""
This module does the following:
Exports the model folder to generate a Model Archive file out of it in .mar format
"""
from . import version

__version__ = version.__version__

from .model_archiver import ModelArchiver
from .model_archiver_config import ModelArchiverConfig
