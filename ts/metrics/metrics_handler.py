"""
Metrics Handler is a custom model handler to test MetricCache changes.
"""

import time

from ts.torch_handler.base_handler import BaseHandler
from ts.service import emit_metrics


class MetricsHandler(BaseHandler):
    """
    Custom model handler for MetricsCache object
    """

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics
        time.sleep(3)
        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        metrics.add_counter(
            "HandlerCounter", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        metrics.add_counter(
            "HandlerCounter", 2.5, None, "ms"
        )
        metrics.add_counter(
            "HandlerCounter", -1.3, None, "ms"
        )
        emit_metrics(metrics.cache)
