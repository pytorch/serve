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

        # "add_metric_to_cache" will only register/override(if already present) a metric object in the metric cache and will not emit it
        self.inf_request_count = metrics.add_metric_to_cache(
            metric_name="InferenceRequestCount",
            unit="count",
            dimension_names=[],
            metric_type=MetricTypes.COUNTER,
        )
        metrics.add_metric_to_cache(
            metric_name="PreprocessCallCount",
            unit="count",
            dimension_names=["ModelName"],
            metric_type=MetricTypes.COUNTER,
        )

        # "add_metric" will register the metric if not already present in metric cache,
        # include the "ModelName" and "Level" dimensions by default and emit it
        metrics.add_metric(
            name="InitializeCallCount",
            value=1,
            unit="count",
            dimensions=[
                Dimension(name="ModelName", value=context.model_name),
                Dimension(name="Level", value="Model"),
            ],
            metric_type=MetricTypes.COUNTER,
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

        # "add_or_update" will emit the metric
        self.inf_request_count.add_or_update(value=1, dimension_values=[])

        # "get_metric" will fetch the corresponding metric from metric cache if present
        preprocess_call_count_metric = metrics.get_metric(
            metric_name="PreprocessCallCount", metric_type=MetricTypes.COUNTER
        )
        preprocess_call_count_metric.add_or_update(
            value=1, dimension_values=[self.context.model_name]
        )

        request_batch_size_metric = metrics.get_metric(
            metric_name="RequestBatchSize", metric_type=MetricTypes.GAUGE
        )
        request_batch_size_metric.add_or_update(
            value=len(data), dimension_values=[self.context.model_name]
        )

        input = data[0].get("body")

        # "add_size" will register the metric if not already present in metric cache,
        # include the "ModelName" and "Level" dimensions by default and emit it
        metrics.add_size(
            name="SizeOfImage", value=len(input) / 1024, idx=None, unit="kB"
        )

        preprocessed_image = ImageClassifier.preprocess(self, data)

        preprocess_stop = time.time()

        # "add_time" will register the metric if not already present in metric cache,
        # include the "ModelName" and "Level" dimensions by default and emit it
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
        # "add_counter" will register the metric if not already present in metric cache,
        # include the "ModelName" and "Level" dimensions by default and emit it
        self.context.metrics.add_counter(
            name="PostprocessCallCount", value=1, idx=None, dimensions=[]
        )
        # "add_percent" will register the metric if not already present in metric cache,
        # include the "ModelName" and "Level" dimensions by default and emit it
        self.context.metrics.add_percent(
            name="ExamplePercentMetric",
            value=50,
            idx=None,
            dimensions=[],
            metric_type=MetricTypes.HISTOGRAM,
        )

        return data.argmax(1).flatten().tolist()
