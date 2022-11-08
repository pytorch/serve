"""
Abstract metric cache class using ABC.
Implemented in the case it is decided that another file format is better in the future.

Currently, abstract class has the methods for getting a metric and adding a metric to the cache.
"""
import abc
import ts.metrics.metric_cache_errors as merrors

from ts.metrics.dimension import Dimension
from ts.metrics.imetric import IMetric
from ts.metrics.metric_type_enum import MetricTypes


class MetricCacheAbstract(metaclass=abc.ABCMeta):
    def __init__(self, request_ids, model_name, config_file_path):
        """
        Constructor for MetricsCaching class

            MetricsCaching class will hold Metric objects in a cache that can be edited and read from.
            The class also has the ability to parse a yaml file into Metrics objects

        Parameters
        ----------
        config_file_path: str
            Name of file to be parsed

        """
        self.cache = dict()
        self.store = []
        self.request_ids = request_ids
        self.model_name = model_name
        self.config_file_path = config_file_path

    def _update_dims(self, idx, dimensions):
        dim_names = [dim.name for dim in dimensions]
        if idx is None:
            if "Level" not in dim_names:
                dimensions.append(Dimension("Level", "Error"))
        else:
            if "ModelName" not in dim_names:
                dimensions.append(Dimension("ModelName", self.model_name))
            if "Level" not in dim_names:
                dimensions.append(Dimension("Level", "Model"))
        return dimensions

    def add_counter(
        self,
        metric_name: str,
        value: int or float,
        idx: str = None,
        dimensions: list = [],
    ):
        """
        Add a counter metric or increment an existing counter metric
            Default metric type is counter
        Parameters
        ----------
        metric_name: str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        dimensions: list
            list of dimension names for the metric
        """
        dimensions = self._update_dims(idx, dimensions)
        metric = self.add_metric(
            metric_name=metric_name,
            unit="count",
            metric_type=MetricTypes.COUNTER,
            dimension_names=[dim.name for dim in dimensions],
        )
        metric.add_or_update(value, [dim.value for dim in dimensions], idx)

    def add_time(
        self,
        metric_name: str,
        value: int or float,
        idx: str = None,
        unit: str = "ms",
        dimensions: list = [],
        metric_type: MetricTypes = MetricTypes.GAUGE,
    ):
        """
        Add a time based metric like latency, default unit is 'ms'
            Default metric type is gauge

        Parameters
        ----------
        metric_name : str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        unit: str
            unit of metric,  default here is ms, s is also accepted
        dimensions: list
            list of dimension names for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        if unit not in ["ms", "s"]:
            raise merrors.MetricsCacheValueError(
                "the unit for a timed metric should be one of ['ms', 's']"
            )
        dimensions = self._update_dims(idx, dimensions)
        metric = self.add_metric(
            metric_name=metric_name,
            unit=unit,
            metric_type=metric_type,
            dimension_names=[dim.name for dim in dimensions],
        )
        metric.add_or_update(value, [dim.value for dim in dimensions], idx)

    def add_size(
        self,
        metric_name: str,
        value: int or float,
        idx: str = None,
        unit: str = "MB",
        dimensions: list = [],
        metric_type: MetricTypes = MetricTypes.GAUGE,
    ):
        """
        Add a size based metric
            Default metric type is gauge

        Parameters
        ----------
        metric_name : str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        unit: str
            unit of metric, default here is 'MB', 'kB', 'GB' also supported
        dimensions: list
            list of dimensions for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        if unit not in ["MB", "kB", "GB", "B"]:
            raise ValueError(
                "The unit for size based metric is one of ['MB','kB', 'GB', 'B']"
            )
        dimensions = self._update_dims(idx, dimensions)
        metric = self.add_metric(
            metric_name=metric_name,
            unit=unit,
            metric_type=metric_type,
            dimension_names=[dim.name for dim in dimensions],
        )
        metric.add_or_update(value, [dim.value for dim in dimensions], idx)

    def add_percent(
        self,
        metric_name: str,
        value: int or float,
        idx: str = None,
        dimensions: list = [],
        metric_type: MetricTypes = MetricTypes.GAUGE,
    ):
        """
        Add a percentage based metric
            Default metric type is gauge

        Parameters
        ----------
        metric_name : str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        dimensions: list
            list of dimensions for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        dimensions = self._update_dims(idx, dimensions)
        metric = self.add_metric(
            metric_name=metric_name,
            unit="percent",
            metric_type=metric_type,
            dimension_names=[dim.name for dim in dimensions],
        )
        metric.add_or_update(value, [dim.value for dim in dimensions], idx)

    def add_error(
        self,
        metric_name: str,
        value: int or float,
        dimensions: list = [],
    ):
        """
        Add an Error Metric
            Default metric type is counter

        Parameters
        ----------
        metric_name : str
            metric name
        value: int or float
            value of the metric
        dimensions: list
            list of dimension objects for the metric
        """
        dimensions = self._update_dims(None, dimensions)
        metric = self.add_metric(
            metric_name=metric_name,
            unit="",
            metric_type=MetricTypes.COUNTER,
            dimension_names=[dim.name for dim in dimensions],
        )
        metric.add_or_update(value, [dim.value for dim in dimensions])

    def get_metric(
        self,
        metric_name: str,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> IMetric:
        """
        Create a new metric and add into cache

        Parameters
        ----------
        metric_name str
            Name of metric

        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram

        Returns
        -------
        Metrics object or MetricsCacheKeyError if not found
        """
        pass

    def add_metric(
        self,
        metric_name: str,
        unit: str,
        dimension_names: list = None,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> IMetric:
        """
        Create a new metric and add into cache.
            Add a metric which is generic with custom metrics

        Parameters
        ----------
        metric_name: str
            Name of metric
        unit: str
            unit of metric
        dimension_names: list
            list of dimensions for the metric
        metric_type: MetricTypes
            Type of metric

        Returns
        -------
        Created metrics object
        """
        pass
