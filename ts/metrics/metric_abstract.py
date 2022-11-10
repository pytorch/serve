"""
Interface for metric class for TS
"""
import abc
from ts.metrics.unit import Units
from ts.metrics.metric_type_enum import MetricTypes

MetricUnit = Units()


class MetricAbstract(metaclass=abc.ABCMeta):
    """
    Class for generating metrics and printing it to stdout of the worker
    """

    def __init__(
        self,
        metric_name: str,
        unit: str,
        dimension_names: list = None,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ):
        """
        Constructor for CachingMetric
           CachingMetric reports collected metrics to stdout of worker

        Parameters
        ----------
        metric_name str
            Name of metric

        unit str
            unit can be one of ms, percent, count, MB, GB or a generic string

        dimension_names list
            list of dimension names which should be strings

        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram

        """
        self.metric_name = metric_name
        if unit in list(MetricUnit.units.keys()):
            self.unit = MetricUnit.units[unit]
        self.dimension_names = dimension_names or []
        self.metric_type = metric_type

    @abc.abstractmethod
    def add_or_update(
        self,
        value: int or float,
        dimension_values: list,
        request_id: str = "",
    ):
        pass
