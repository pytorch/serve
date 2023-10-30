"""
Command line interface to export model files to be used for inference by MXNet Model Server
"""

from model_archiver.model_archiver_config import ModelArchiverConfig
from model_archiver.model_packaging import generate_model_archive


class ModelArchiver:
    @staticmethod
    def generate_model_archive(config: ModelArchiverConfig) -> None:
        """
        Generate a model archive file
        :param config: Model Archiver Config object
        :return:
        """
        generate_model_archive(config)
