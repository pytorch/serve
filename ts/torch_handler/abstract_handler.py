import abc
import logging
import time

from ts.handler_utils.timer import timed

logger = logging.getLogger(__name__)


class AbstractHandler(abc.ABC):
    """
    Base default handler to load the model
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        self.device = None
        self.context = None
        self.explain = False

    @abc.abstractmethod
    def initialize(self, context):
        """
        Initialize function loads the model and initializes the model object.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        pass

    @abc.abstractmethod
    def as_tensor(self, data):
        """
        Convert data to tensor consumable by the underlying model.
        Used for preprocessing the request.

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        pass

    @timed
    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        return self.as_tensor(data)

    @abc.abstractmethod
    @timed
    def inference(self, data, *args, **kwargs):
        pass

    @timed
    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (numpy array-like structure): The tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """

        return data.tolist()

    def handle(self, data, context):
        """
        Entry point for default handler. It takes the data from the input request and returns
        the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artifacts parameters.

        Returns:
            list: Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = self.profiler_enabled()
        if is_profiler_enabled:
            output, _ = self.infer_with_profiler(data=data, context=context)
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output

    def explain_handle(self, data_preprocess, raw_data):
        """
        Captum explanations handler

        Args:
            data_preprocess (numpy array-like structure): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request

        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = None
        inputs = None
        target = 0

        logger.info("Calculating Explanations")
        row = raw_data[0]
        if isinstance(row, dict):
            logger.info("Getting data and target")
            inputs = row.get("data") or row.get("body")
            target = row.get("target")
            if not target:
                target = 0

        output_explain = self.get_insights(data_preprocess, inputs, target)
        return output_explain

    @abc.abstractmethod
    def get_insights(self, tensor_data, _, target=0):
        pass

    def _is_explain(self):
        if self.context and self.context.get_request_header(0, "explain"):
            if self.context.get_request_header(0, "explain") == "True":
                self.explain = True
                return True
        return False

    def _is_describe(self):
        if self.context and self.context.get_request_header(0, "describe"):
            if self.context.get_request_header(0, "describe") == "True":
                return True
        return False

    @abc.abstractmethod
    def describe_handle(self):
        """Customized describe handler

        Returns:
            dict : A dictionary response.
        """
        pass

    def get_device(self):
        """Get device

        Returns:
            string : self device
        """
        return self.device

    @abc.abstractmethod
    def profiler_enabled(self):
        """
        Return true if profiler is enabled

        Returns:
            bool: true if profiler is enabled
        """
        pass

    @abc.abstractmethod
    def infer_with_profiler(self, data, context):
        """
        Custom method to for handling the inference with profiler

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artifacts parameters.

        Returns:
            output : Returns a list of dictionary with the predicted response.
            prof: profiler object
        """
        pass
