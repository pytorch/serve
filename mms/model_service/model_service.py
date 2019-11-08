

"""`ModelService` defines an API for base model service.
"""
# pylint: disable=W0223

import ast
import json
import logging
import os
import time
from abc import ABCMeta, abstractmethod


class ModelService(object):
    """
    ModelService wraps up all preprocessing, inference and postprocessing
    functions used by model service. It is defined in a flexible manner to
    be easily extended to support different frameworks.
    """
    __metaclass__ = ABCMeta

    # noinspection PyUnusedLocal
    def __init__(self, model_name, model_dir, manifest, gpu=None):  # pylint: disable=unused-argument
        self.ctx = None
        self._context = None
        self._signature = None

    def initialize(self, context):
        """
        Internal initialize ModelService.

        :param context: MMS context object
        :return:
        """
        self._context = context
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        signature_file_path = os.path.join(model_dir, context.manifest['Model']['Signature'])
        if not os.path.isfile(signature_file_path):
            raise ValueError("Signature file is not found.")

        with open(signature_file_path) as f:
            self._signature = json.load(f)

    @abstractmethod
    def inference(self, data):
        """
        Wrapper function to run pre-process, inference and post-process functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        """
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def ping(self):
        """
        Ping to get system's health.

        Returns
        -------
        String
            A message, "health": "healthy!", to show system is healthy.
        """
        # pylint: disable=unnecessary-pass
        pass

    def signature(self):
        """
        Signature for model service.

        Returns
        -------
        Dict
            Model service signature.
        """
        return self._signature

    # noinspection PyUnusedLocal
    def handle(self, data, context):  # pylint: disable=unused-argument
        """
        Backward compatible handle function.

        :param data:
        :param context:
        :return:

        """
        input_type = self._signature['input_type']

        input_data = []
        data_name = self._signature["inputs"][0]["data_name"]
        form_data = data[0].get(data_name)
        if form_data is None:
            form_data = data[0].get("body")

        if form_data is None:
            form_data = data[0].get("data")

        if input_type == "application/json":
            # user might not send content in HTTP request
            if isinstance(form_data, (bytes, bytearray)):
                form_data = ast.literal_eval(form_data.decode("utf-8"))

        input_data.append(form_data)

        ret = self.inference(input_data)
        if isinstance(ret, list):
            return ret

        return [ret]


class SingleNodeService(ModelService):
    """
    SingleNodeModel defines abstraction for model service which loads a
    single model.
    """

    def inference(self, data):
        """
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        """
        preprocess_start = time.time()
        data = self._preprocess(data)
        inference_start = time.time()
        data = self._inference(data)
        postprocess_start = time.time()
        data = self._postprocess(data)
        end_time = time.time()

        logging.info("preprocess time: %.2f", (inference_start - preprocess_start) * 1000)
        logging.info("inference time: %.2f", (postprocess_start - inference_start) * 1000)
        logging.info("postprocess time: %.2f", (end_time - postprocess_start) * 1000)

        return data

    @abstractmethod
    def _inference(self, data):
        """
        Internal inference methods. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        """
        return data

    def _preprocess(self, data):
        """
        Internal preprocess methods. Do transformation on raw
        inputs and convert them to NDArray.

        Parameters
        ----------
        data : list of object
            Raw inputs from request.

        Returns
        -------
        list of NDArray
            Processed inputs in NDArray format.
        """
        return data

    def _postprocess(self, data):
        """
        Internal postprocess methods. Do transformation on inference output
        and convert them to MIME type objects.

        Parameters
        ----------
        data : list of NDArray
            Inference output.

        Returns
        -------
        list of object
            list of outputs to be sent back.
        """
        return data
