"""
Abstract metric cache class using ABC.
Implemented in the case it is decided that another file format is better in the future.

Currently, abstract class has the methods for getting a metric and adding a metric to the cache.
"""
import abc
import logging
import ts.metrics.metric_cache_errors as merrors

from ts.metrics.metric import Metric
from ts.metrics.dimension import Dimension


class MetricCacheAbstract(metaclass=abc.ABCMeta):
    def __init__(self, file, model_name):
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
        self.model_name = model_name

    def _add_or_update(self, name, value, unit, metric_type=None, dimensions=None, error_metric_bool=False):
        """
        Add a metric key value pair

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        unit: str
            unit of metric
        value: int, float, str
            value of metric
        metric_type: str, optional
            indicates type of metric operation if it is defined, creates default metric type if not defined
        dimensions: list, optional
            list of Dimension objects
        """
        # IF req_id is none error Metric
        if dimensions is None:
            dimensions = list()
        elif not isinstance(dimensions, list):
            raise merrors.MetricsCacheValueError("Please provide a list of Dimension objects.")
        if error_metric_bool:
            dimensions.append(Dimension("Level", "Error"))
        else:
            dimensions.append(Dimension("ModelName", self.model_name))
            dimensions.append(Dimension("Level", "Model"))

        # Cache the metric with a unique key
        dims_str = ",".join([str(d) for d in dimensions])
        metric_key = f"[{metric_type}]-[{name}]-[{dims_str}]"

        if metric_key not in self.cache:
            self.add_metric(metric_name=name, unit=unit, dimensions=dimensions, metric_type=metric_type, value=value)
        else:
            existing_metric = self.get_metric(metric_key=metric_key)
            existing_metric.update(value)

    def add_counter(self, name, value, idx=None, dimensions=None, metric_type="counter"):
        """
        Add a counter metric or increment an existing counter metric
            Default metric type is counter
        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        metric_type: str
           type for defining different operations, defaulted to counter metric type for Counter metrics
        """
        unit = 'count'
        self._add_or_update(name=name, value=value, unit=unit, metric_type=metric_type, dimensions=dimensions)

    def add_time(self, name, value, idx=None, unit='ms', dimensions=None, metric_type="gauge"):
        """
        Add a time based metric like latency, default unit is 'ms'
            Default metric type is gauge

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric,  default here is ms, s is also accepted
        dimensions: list
            list of dimensions for the metric
        metric_type: str
           type for defining different operations, defaulted to gauge metric type for Time metrics
        """
        if unit not in ['ms', 's']:
            raise merrors.MetricsCacheValueError("the unit for a timed metric should be one of ['ms', 's']")
        self._add_or_update(name=name, value=value, unit=unit, metric_type=metric_type, dimensions=dimensions)

    def add_size(self, name, value, idx=None, unit='MB', dimensions=None, metric_type="gauge"):
        """
        Add a size based metric
            Default metric type is gauge

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric, default here is 'MB', 'kB', 'GB' also supported
        dimensions: list
            list of dimensions for the metric
        metric_type: str
           type for defining different operations, defaulted to gauge metric type for Size metrics
        """
        if unit not in ['MB', 'kB', 'GB', 'B']:
            raise ValueError("The unit for size based metric is one of ['MB','kB', 'GB', 'B']")
        self._add_or_update(name=name, value=value, unit=unit, metric_type=metric_type, dimensions=dimensions)

    def add_percent(self, name, value, idx=None, dimensions=None, metric_type="gauge"):
        """
        Add a percentage based metric
            Default metric type is gauge

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        metric_type: str
           type for defining different operations, defaulted to gauge metric type for Percent metrics
        """
        unit = 'percent'
        self._add_or_update(name=name, value=value, unit=unit, metric_type=metric_type, dimensions=dimensions)

    def add_error(self, name, value, dimensions=None, metric_type="counter"):
        """
        Add an Error Metric
            Default metric type is counter

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric, in this case a str
        dimensions: list
            list of dimensions for the metric
        metric_type: str
           type for defining different operations, defaulted to counter metric type for Error metrics
        """
        unit = ''
        # noinspection PyTypeChecker
        self._add_or_update(name=name, value=value, unit=unit, metric_type=metric_type, dimensions=dimensions,
                            error_metric_bool=True)

    def get_metric(self, metric_key: str) -> Metric:
        """
        Get a metric from cache

        Parameters
        ----------
        metric_key: str
            Key to identify a Metric object within the cache

        """
        if not isinstance(metric_key, str):
            raise merrors.MetricsCacheTypeError(f"Only string types are acceptable as argument.")

        logging.info(f"Getting metric {metric_key}")
        metric_obj = self.cache.get(metric_key)
        if metric_obj:
            logging.debug(f"Successfully received metric {metric_key}")
            return metric_obj
        else:
            raise merrors.MetricsCacheKeyError(f"Metric key {metric_key} does not exist.")

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
                isinstance(metric_type, str) or not isinstance(value, (float, int)):
            raise merrors.MetricsCacheTypeError(f"metric_name must be a str, unit must be a str, "
                                                f"dimensions must be a list of Dimension objects, "
                                                f"metric type must be a str, value must be a int/float")

        if not isinstance(dimensions[0], Dimension):
            raise merrors.MetricsCacheTypeError(f"Dimensions list is expected to be made up of Dimension objects.")

        # Make sure that the passed arguments follow a valid naming convention (doesn't contain certain characters)
        self._inspect_naming_convention(metric_name, unit, dimensions, metric_type, value)

        logging.debug(f"Adding metric with fields of: metric name - {metric_name}, unit - {unit}, "
                      f"dimensions - {dimensions}, metric type - {metric_type}")

        dims_str = ",".join([str(d) for d in dimensions])
        self.cache[f"[{metric_type}]-[{metric_name}]-[{dims_str}]"] = Metric(name=metric_name,
                                                                             value=value,
                                                                             unit=unit,
                                                                             dimensions=dimensions,
                                                                             metric_type=metric_type)

        logging.info(f"Successfully added {metric_name} Metric object to cache.")

    @staticmethod
    def _inspect_naming_convention(*metric_arg) -> None:
        """
        Inspect naming convention for each argument being used to create a Metric object.

            Checking to ensure that certain symbols (- / []) are not used in args so that Metric strings can be created without
            ambiguity.

        Parameters
        ----------
        metric_arg: str/list
        """

        def _check_individual_arg(delimiters: list, arg: str) -> None:
            """
            Checking and validating an individual argument
            """
            for delim in delimiters:
                if delim in str(arg):
                        logging.warning(f"There is a '{delim}' symbol found in {arg} argument. "
                                        f"Please refrain from using the "
                                        f"'{delim}' as it is used as the delimiter in the Metric object string.")

        delimiters = ["-", "[", "]"]  # list of symbols that should not be included in any Metric arguments

        for individual_metric_arg in metric_arg:
            if isinstance(individual_metric_arg, list):  # list of Dimension objects
                for dimension in individual_metric_arg:
                    _check_individual_arg(delimiters, dimension.__str__())
            else:  # should always be every other Metric argument (and should all be string parsable)
                _check_individual_arg(delimiters, individual_metric_arg)
