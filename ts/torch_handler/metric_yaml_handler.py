"""
Custom Handler for parsing model metrics from yaml input
"""

from ts.torch_handler.base_handler import BaseHandler


class ModelMetricYamlHandler(BaseHandler):
    """
    Class for handling model metrics that are being parsed via yaml file
    """

    def preprocess(self, data: list):
        print("Hello")


if __name__ == "__main__":
    handler = ModelMetricYamlHandler()
    handler.preprocess(["safe"])

