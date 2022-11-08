"""
Metric class for model server
"""
import socket
import time
from builtins import str
from collections import OrderedDict
from ts.metrics.caching_metric import CachingMetric
from ts.metrics.metric_type_enum import MetricTypes

from ts.metrics.unit import Units

MetricUnit = Units()


class Metric(object):
    """
    Class for generating metrics and printing it to stdout of the worker
    """

    def __init__(
        self,
        name,
        value,
        unit,
        dimensions,
        request_id=None,
        metric_method=None,
    ):
        """
        Constructor for Metric class

           Metric class will spawn a thread and report collected metrics to stdout of worker

        Parameters
        ----------
        name: str
            Name of metric
        value : int, float
           Can be integer or float
        unit: str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimensions: list
            list of dimension objects
        request_id: str
            req_id of metric
        metric_method: str
           useful for defining different operations, optional

        """
        self.metric_type = MetricTypes.COUNTER
        self.dimensions = dimensions
        self.dimension_names = [dim.name for dim in dimensions]
        self.dimension_values = [dim.value for dim in dimensions]
        self._caching_metric = CachingMetric(
            metric_name=name,
            unit=unit,
            dimension_names=self.dimension_names,
            metric_type=self.metric_type,
        )
        self.name = self._caching_metric.metric_name
        self.unit = self._caching_metric.unit
        self.metric_method = metric_method
        self.value = value
        self.request_id = request_id

    def update(self, value):
        """
        Update function for Metric class

        Parameters
        ----------
        value : int, float
            metric to be updated
        """
        self._caching_metric.add_or_update(value, self.dimension_values, request_id=self.request_id)

    def reset(self):
        """
        Reset Metric value to 0
        """
        self._caching_metric.add_or_update(0, self.dimension_values, request_id=self.request_id)

    def __str__(self):
        dims = ",".join([str(d) for d in self.dimensions])
        if self.request_id:
            return (
                f"{self.name}.{self.unit}:{self.value}|#{dims}|#hostname:{socket.gethostname()},"
                f"{int(time.time())},{self.request_id}"
            )

        return f"{self.name}.{self.unit}:{self.value}|#{dims}|#hostname:{socket.gethostname()},{int(time.time())}"

    def to_dict(self):
        """
        return an Ordered Dictionary
        """
        return OrderedDict(
            {
                "MetricName": self.name,
                "Value": self.value,
                "Unit": self.unit,
                "Dimensions": self.dimensions,
                "Timestamp": int(time.time()),
                "HostName": socket.gethostname(),
                "RequestId": self.request_id,
            }
        )
