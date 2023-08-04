import time

from torchvision import transforms

from ts.metrics.dimension import Dimension
from ts.metrics.metric_type_enum import MetricTypes
from ts.torch_handler.image_classifier import ImageClassifier


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.
    """

    image_processing = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    def initialize(self, context):
        super().initialize(context)
        metrics = context.metrics

        # Usage of "add_metric"
        self.inf_request_count = metrics.add_metric(
            metric_name="InferenceRequestCount",
            unit="count",
            dimension_names=[],
            metric_type=MetricTypes.COUNTER,
        )
        metrics.add_metric(
            metric_name="RequestBatchSize",
            unit="count",
            dimension_names=["ModelName"],
            metric_type=MetricTypes.GAUGE,
        )

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        preprocess_start = time.time()

        metrics = self.context.metrics

        # Usage of "add_or_update"
        self.inf_request_count.add_or_update(value=1, dimension_values=[])

        # Usage of "get_metric"
        request_batch_size_metric = metrics.get_metric(
            metric_name="RequestBatchSize", metric_type=MetricTypes.GAUGE
        )
        request_batch_size_metric.add_or_update(
            value=len(data), dimension_values=[self.context.model_name]
        )

        input = data[0].get("body")

        # Usage of "add_size"
        metrics.add_size(
            name="SizeOfImage", value=len(input) / 1024, idx=None, unit="kB"
        )

        preprocessed_image = ImageClassifier.preprocess(self, data)

        preprocess_stop = time.time()

        # usage of add_time
        metrics.add_time(
            name="HandlerMethodTime",
            value=(preprocess_stop - preprocess_start) * 1000,
            idx=None,
            unit="ms",
            dimensions=[Dimension(name="MethodName", value="preprocess")],
            metric_type=MetricTypes.GAUGE,
        )

        return preprocessed_image

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionary with predictons and explanations are returned.
        """
        # usage of add_counter
        self.context.metrics.add_counter(
            name="PostprocessCallCount", value=1, idx=None, dimensions=[]
        )
        # usage of add_percent
        self.context.metrics.add_percent(
            name="ExamplePercentMetric",
            value=50,
            idx=None,
            dimensions=[],
            metric_type=MetricTypes.HISTOGRAM,
        )

        return data.argmax(1).flatten().tolist()
