"""
Caching metric class for TS
"""
import logging
import socket
import time

from ts.metrics.dimension import Dimension
from ts.metrics.metric_abstract import MetricAbstract
from ts.metrics.metric_type_enum import MetricTypes
logger = logging.getLogger(__name__)


class CachingMetric(MetricAbstract):
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
        super().__init__(metric_name, unit, dimension_names, metric_type)

    def _validate_and_get_dimensions(
        self,
        dimension_values: list,
    ) -> list:
        """
        Validates that the dimension values match the dimension names
        amd creates dimension objs
        Parameters
        ----------
        dimension_values
            values corresponding to the metrics dimension names
        Returns
        -------
        list of dimension objects or ValueError
        """
        if dimension_values is None or len(dimension_values) != len(self.dimension_names):
            raise ValueError(
                f"Dimension values: {dimension_values} should "
                f"correspond to Dimension names: {self.dimension_names}"
            )
        dim_objs = []
        for dim_name, dim_value in zip(self.dimension_names, dimension_values):
            dim_objs.append(Dimension(dim_name, dim_value))
        return dim_objs

    def _validate_metric_value(
        self,
        value: int or float,
    ) -> None:
        """
        Validations for metric value
        Parameters
        ----------
        value
            metric value to be updated
        """
        if self.metric_type == MetricTypes.COUNTER and value < 0:
            raise ValueError("Counter metric update value cannot be negative")

    def emit_metrics(
        self,
        request_id: str,
        value: int or float,
        dimension_string: str,
    ) -> None:
        """
        Logs metrics to
        Parameters
        ----------
        request_id
        value
        dimension_string
        """
        metric_str = f"[METRICS]{self.metric_name}.{self.unit}:{value}|#{dimension_string}|" \
                     f"#hostname:{socket.gethostname()},{int(time.time())}"
        if request_id:
            logger.info(f"{metric_str},{request_id}")
        else:
            logger.info(metric_str)

    def add_or_update(
        self,
        value: int or float,
        dimension_values: list = [],
        request_id: str = "",
    ):
        """
        Update metric value, request id and dimensions

        Parameters
        ----------
        value : int, float
            metric to be updated
        dimension_values : list
            list of dimension values
        request_id : str
            request id to be associated with the metric
        """
        dimension_str = ""
        try:
            dimension_objs = self._validate_and_get_dimensions(dimension_values)
            dimension_str = ",".join([str(d) for d in dimension_objs])
            self._validate_metric_value(value)
        except ValueError as ex:
            logger.error(
                f"[METRICS]Failed to update metric with name:"
                f"{self.metric_name} and dimensions: {dimension_str} "
                f"with value: {value}: {ex}"
            )
        else:
            self.emit_metrics(request_id, value, dimension_str)

    def update(
        self,
        value: int or float,
        request_id: str = "",
        dimensions: list = [],
    ):
        """
        BACKWARDS COMPATIBILITY: Update metric value

        Parameters
        ----------
        value : int, float
            metric to be updated
        request_id : str
            request id to be associated with the metric
        dimensions : list
            list of dimension values
        """
        logger.warning("Overriding existing dimensions")
        self.dimension_names = [dim.name for dim in dimensions]
        self.add_or_update(value, [dim.value for dim in dimensions], request_id)
