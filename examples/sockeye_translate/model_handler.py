

"""
ModelHandler defines a base model handler.
"""
import logging


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.

        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        return None

    def inference(self, model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        return None

    def postprocess(self, inference_output):
        """
        Return predict result in batch.

        :param inference_output: list of inference output
        :return: list of predict results
        """
        return ["OK"] * self._batch_size

    def handle(self, data, context):
        """
        Custom service entry point function.

        :param data: list of objects, raw input from request
        :param context: model server context
        :return: list of outputs to be send back to client
        """

        try:
            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)

            return data
        except Exception as e:
            logging.error(e, exc_info=True)
            request_processor = context.request_processor
            request_processor.report_status(500, "Unknown inference error")
            return [str(e)] * self._batch_size
