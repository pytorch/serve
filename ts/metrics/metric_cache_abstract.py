"""
Abstract metric cache class using ABC.
Implemented in the case it is decided that another file format is better in the future.

Currently, abstract class has the methods for getting a metric and adding a metric to the cache.
"""
import abc
import os

import ts.metrics.metric_cache_errors as merrors
from ts.metrics.dimension import Dimension
from ts.metrics.metric_abstract import MetricAbstract
from ts.metrics.metric_type_enum import MetricTypes


class MetricCacheAbstract(metaclass=abc.ABCMeta):
    def __init__(self, config_file_path):
        """
        Constructor for MetricsCaching class

            MetricsCaching class will hold Metric objects in a cache that can be edited and read from.
            The class also has the ability to parse a yaml file into Metrics objects

        Parameters
        ----------
        config_file_path: str
            Name of file to be parsed

        """
        self.cache = {}
        self.store = []
        self.request_ids = None
        self.model_name = None
        self.config_file_path = config_file_path
        try:
            os.path.exists(self.config_file_path)
        except Exception as exc:
            raise merrors.MetricsCacheTypeError(
                f"Error loading {config_file_path} file: {exc}"
            )

    def _add_default_dims(self, idx, dimensions):
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

    def set_request_ids(self, request_ids):
        self.request_ids = request_ids

    def _get_req(self, idx):
        """
        Provide the request id dimension
        Parameters
        ----------
        idx : int
            request_id index in batch
        """
        # check if request id for the metric is given, if so use it else have a list of all.
        req_id = self.request_ids
        if isinstance(req_id, dict):
            req_id = ",".join(self.request_ids.values())
        if idx is not None and self.request_ids is not None and idx in self.request_ids:
            req_id = self.request_ids[idx]
        return req_id

    def add_metric(
        self,
        name: str,
        value: int or float,
        unit: str,
        idx: str = None,
        dimensions: list = [],
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ):
        """
        Add a generic metric
            Default metric type is counter

        Parameters
        ----------
        name : str
            metric name
        value: int or float
            value of the metric
        unit: str
            unit of metric
        idx: str
            request id to be associated with the metric
        dimensions: list
            list of Dimension objects for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        req_id = self._get_req(idx)
        dimensions = self._add_default_dims(req_id, dimensions)
        metric = self._get_or_add_metric(name, unit, dimensions, metric_type)
        metric.add_or_update(value, [dim.value for dim in dimensions], req_id)

    def add_counter(
        self,
        name: str,
        value: int or float,
        idx: str = None,
        dimensions: list = [],
    ):
        """
        Add a counter metric or increment an existing counter metric
            Default metric type is counter
        Parameters
        ----------
        name: str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        dimensions: list
            list of Dimension objects for the metric
        """
        req_id = self._get_req(idx)
        dimensions = self._add_default_dims(req_id, dimensions)
        metric = self._get_or_add_metric(name, "count", dimensions, MetricTypes.COUNTER)
        metric.add_or_update(value, [dim.value for dim in dimensions], req_id)

    def add_time(
        self,
        name: str,
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
        name : str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        unit: str
            unit of metric,  default here is ms, s is also accepted
        dimensions: list
            list of Dimension objects for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        if unit not in ["ms", "s"]:
            raise merrors.MetricsCacheValueError(
                "the unit for a timed metric should be one of ['ms', 's']"
            )
        req_id = self._get_req(idx)
        dimensions = self._add_default_dims(req_id, dimensions)
        metric = self._get_or_add_metric(name, unit, dimensions, metric_type)
        metric.add_or_update(value, [dim.value for dim in dimensions], req_id)

    def add_size(
        self,
        name: str,
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
        name : str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        unit: str
            unit of metric, default here is 'MB', 'kB', 'GB' also supported
        dimensions: list
            list of Dimension objects for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        if unit not in ["MB", "kB", "GB", "B"]:
            raise ValueError(
                "The unit for size based metric is one of ['MB','kB', 'GB', 'B']"
            )
        req_id = self._get_req(idx)
        dimensions = self._add_default_dims(req_id, dimensions)
        metric = self._get_or_add_metric(name, unit, dimensions, metric_type)
        metric.add_or_update(value, [dim.value for dim in dimensions], req_id)

    def add_percent(
        self,
        name: str,
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
        name : str
            metric name
        value: int or float
            value of the metric
        idx: str
            request id to be associated with the metric
        dimensions: list
            list of Dimension objects for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
        req_id = self._get_req(idx)
        dimensions = self._add_default_dims(req_id, dimensions)
        metric = self._get_or_add_metric(name, "percent", dimensions, metric_type)
        metric.add_or_update(value, [dim.value for dim in dimensions], req_id)

    def add_error(
        self,
        name: str,
        value: int or float,
        dimensions: list = [],
    ):
        """
        Add an Error Metric
            Default metric type is counter

        Parameters
        ----------
        name : str
            metric name
        value: int or float
            value of the metric
        dimensions: list
            list of Dimension objects for the metric
        """
        dimensions = self._add_default_dims(None, dimensions)
        metric = self._get_or_add_metric(name, "", dimensions, MetricTypes.COUNTER)
        metric.add_or_update(value, [dim.value for dim in dimensions])

    def _get_or_add_metric(
        self, metric_name, unit, dimensions, metric_type
    ) -> MetricAbstract:
        try:
            metric = self.get_metric(metric_name, metric_type)
        except merrors.MetricsCacheKeyError:
            metric = self.add_metric_to_cache(
                metric_name=metric_name,
                unit=unit,
                metric_type=metric_type,
                dimension_names=[dim.name for dim in dimensions],
            )
        return metric

    @staticmethod
    def _check_type(variable, expected_type, helper_text):
        if not isinstance(variable, expected_type):
            raise merrors.MetricsCacheTypeError(helper_text)

    @abc.abstractmethod
    def initialize_cache(self) -> None:
        """
        Create Metric objects based off of the model_metrics data and add to cache
        """
        pass

    @abc.abstractmethod
    def get_metric(
        self,
        metric_name: str,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> MetricAbstract:
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

    @abc.abstractmethod
    def add_metric_to_cache(
        self,
        metric_name: str,
        unit: str,
        dimension_names: list = [],
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> MetricAbstract:
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
            list of dimension name strings for the metric
        metric_type: MetricTypes
            Type of metric

        Returns
        -------
        Created metrics object
        """
        pass
