"""
Abstract metric cache class using ABC.
Implemented in the case it is decided that another file format is better in the future.

Currently, abstract class has the methods for getting a metric and adding a metric to the cache.
"""
import abc
import sys
import psutil
import logging

from ts.metrics.metric import Metric
from ts.metrics.dimension import Dimension


class MetricCacheAbstract(metaclass=abc.ABCMeta):
    def __init__(self, file):
        """
        Constructor for MetricsCaching class

            MetricsCaching class will hold Metric objects in a cache that can be edited and read from.
            The class also has the ability to parse a yaml file into Metrics objects

        Parameters
        ----------
        file: str
            Name of file to be parsed

        """
        self.cache = {}
        self.file = file

    def get_metric(self, metric_key: str) -> Metric:
        """
        Get a metric from cache

        Parameters
        ----------
        metric_key: str
            Key to identify a Metric object within the cache

        """
        if not isinstance(metric_key, str):
            logging.error(f"Only string types are acceptable as argument.")
            sys.exit(1)

        logging.info(f"Getting metric {metric_key}")
        metric_obj = self.cache.get(metric_key)
        if metric_obj:
            logging.info("Successfully received metric")
            return metric_obj
        else:
            logging.info("Metric does not exist.")
            sys.exit(1)

    def add_metric(self, metric_name: str, unit: str, dimensions: list, metric_type: str, value=0) -> None:
        """
        Create a new metric and add into cache

        Parameters
        ----------
        metric_name: str
            Name of metric
        unit: str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimensions: list
            list of dimension objects/strings read from yaml file
        metric_type: str
            Type of metric
        value: int, float
            value of metric

        """

        if not isinstance(metric_name, str) or not isinstance(unit, str) or not isinstance(dimensions, list) or not \
                isinstance(metric_type, str):
            raise TypeError(f"metric_name must be a str, unit must be a str, "
                            f"dimensions must be a list of str, metric type must be a str")

        # transforming Dimensions list into list of Dimension objects if not already
        if isinstance(dimensions[0], Dimension):  # this is ideal format
            pass
        # FIXME expecting even number of dimensions list - is this a correct assumption?
        elif len(dimensions) % 2 == 0 and isinstance(dimensions[0], str):
            temp_dimensions = []
            for i in range(len(dimensions)):
                if i % 2 == 0:
                    temp_dimensions.append(Dimension(name=dimensions[i], value=dimensions[i + 1]))
            dimensions = temp_dimensions
        else:
            raise ValueError(f"Dimensions list is expected to be an even number if the list of dimensions"
                             f" is made up of strings.")

        logging.debug(f"Adding metric with fields of: metric name - {metric_name}, unit - {unit}, "
                      f"dimensions - {dimensions}, metric type - {metric_type}")

        dims_str = "-".join([str(d) for d in dimensions])

        self.cache[f"{metric_type}-{metric_name}-{dims_str}"] = Metric(name=metric_name,
                                                                       value=value,
                                                                       unit=unit,
                                                                       dimensions=dimensions,
                                                                       metric_type=metric_type)
        logging.info("Successfully added metric.")

    def _add_all_system_metrics(self):
        """
        Add all system metrics
        """
        dimension = [Dimension('Level', 'Host')]

        # using different dimensions for variation’s sake
        dimension_gpu = [Dimension('Level', 'Host'), Dimension("device_id", "DID")]

        # Adding CPU Utilization metric
        data = psutil.cpu_percent()
        self.add_metric(metric_name="CPUUtilization", value=data, unit="percent",
                        dimensions=dimension, metric_type="CPUUtilizationType")

        # Adding Memory Used metric
        data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
        self.add_metric(metric_name="MemoryUsed", value=data, unit="MB",
                        dimensions=dimension, metric_type="MemoryUsedType")

        # Adding Memory Available metric
        data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
        self.add_metric(metric_name="MemoryAvailable", value=data, unit="MB",
                        dimensions=dimension, metric_type="MemoryAvailableType")

        # Adding Memory Utilization metric
        data = psutil.virtual_memory().percent
        self.add_metric(metric_name="MemoryUtilization", value=data, unit="percent",
                        dimensions=dimension, metric_type="MemoryUtilizationType")

        # Adding Disk Usage metric
        data = psutil.disk_usage('/').used / (1024 * 1024 * 1024)  # in GB
        self.add_metric(metric_name="DiskUsage", value=data, unit="GB",
                        dimensions=dimension, metric_type="DiskUsageType")

        # Adding Disk Utilization metric
        data = psutil.disk_usage('/').percent
        self.add_metric(metric_name="DiskUtilization", value=data, unit="percent",
                        dimensions=dimension, metric_type="DiskUtilizationType")

        # Adding Disk Available metric
        data = psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
        self.add_metric(metric_name="DiskAvailable", value=data, unit="GB",
                        dimensions=dimension_gpu, metric_type="DiskAvailableType")

    def emit_metrics_to_log(self):
        """
        Emit metrics to log statements.
        """
        self._add_all_system_metrics()
        logging.getLogger().setLevel(level=logging.DEBUG)
        for metric_key, metric in self.cache.items():
            logging.info(metric)
