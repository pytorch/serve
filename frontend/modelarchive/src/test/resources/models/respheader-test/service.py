# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
NoopService defines a no operational model handler.
"""
import logging
import time


class NoopService(object):
    """
    Noop Model handler implementation.

    Extend from BaseModelHandler is optional
    """

    def __init__(self):
        self._context = None
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: model server context
        :return:
        """
        self.initialized = True
        self._context = context

    @staticmethod
    def preprocess(data):
        """
        Transform raw input into model input data.

        :param data: list of objects, raw input from request
        :return: list of model input data
        """
        return data

    @staticmethod
    def inference(model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: inference results
        """
        return model_input

    @staticmethod
    def postprocess(model_output):
        return [str(model_output)]

    def handle(self, data, context):
        """
        Custom service entry point function.

        :param context: model server context
        :param data: list of objects, raw input from request
        :return: list of outputs to be send back to client
        """
        # Add your initialization code here
        properties = context.system_properties
        server_name = properties.get("server_name")
        server_version = properties.get("server_version")
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        batch_size = properties.get("batch_size")

        logging.debug("server_name: {}".format(server_name))
        logging.debug("server_version: {}".format(server_version))
        logging.debug("model_dir: {}".format(model_dir))
        logging.debug("gpu_id: {}".format(gpu_id))
        logging.debug("batch_size: {}".format(batch_size))
        request_processor = context.request_processor
        try:
            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)
            context.set_response_content_type(0, "text/plain")

            context.set_response_header(0, "dummy", "1")
            return data
        except Exception as e:
            logging.error(e, exc_info=True)
            request_processor.report_status(500, "Unknown inference error.")
            return ["Error {}".format(str(e))] * len(data)


_service = NoopService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
