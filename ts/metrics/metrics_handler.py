"""
Metrics Handler is a custom model handler to test MetricCache changes.
"""

import time
import logging

from ts.torch_handler.base_handler import BaseHandler
from ts.service import emit_metrics


class MetricsHandler(BaseHandler):
    """
    Custom model handler for MetricsCache object
    """
    def preprocess(self, data):

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()
        logging.warning("IN METRICS CUSTOM HANDLER")

        metrics = self.context.metrics
        time.sleep(3)
        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        metrics.add_counter(
            "HandlerSeparateCounter", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        metrics.add_counter(
            "HandlerSeparateCounter", 2.5, None, "ms"
        )
        metrics.add_counter(
            "HandlerSeparateCounter", -1.3, None, "ms"
        )
        metrics.add_counter(
            "HandlerCounter", -1.3, None, "ms"
        )

        print("IN HERE")
        emit_metrics(metrics.cache)
        return data
