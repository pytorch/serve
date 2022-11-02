"""
Abstract metric cache class using ABC.
Implemented in the case it is decided that another file format is better in the future.

Currently, abstract class has the methods for getting a metric and adding a metric to the cache.
"""
import abc
import logging

import ts.metrics.metric_cache_errors as merrors
from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric
from ts.metrics.metric_type_enum import MetricTypes


class MetricCacheAbstract(metaclass=abc.ABCMeta):
    def __init__(self, request_ids, model_name, config_file):
        """
        Constructor for MetricsCaching class

            MetricsCaching class will hold Metric objects in a cache that can be edited and read from.
            The class also has the ability to parse a yaml file into Metrics objects

        Parameters
        ----------
        config_file: str
            Name of file to be parsed

        """
        self.cache = {}
        self.store = []
        self.request_ids = request_ids
        self.model_name = model_name
        self.config_file = config_file

    @staticmethod
    def _check_matching_dims(key, dims) -> bool:
        """
        Check to see if key value string pair already in list of dim strings
        """
        is_present = False
        for dim in dims:
            key_dim = dim.split(":")[0]
            if key == key_dim:
                is_present = True
        return is_present

    def _add_or_update(
        self,
        name: str,
        value: int or float,
        req_id,
        unit: str,
        metric_type: MetricTypes,
        dimensions: list = None,
    ):
        """
        Add a metric key value pair

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        req_id: str
            request id
        unit: str
            unit of metric
        metric_type: MetricTypes
            indicates type of metric operation if it is defined, creates default metric type if not defined
        dimensions: list, optional
            list of Dimension objects
        """
        # IF req_id is none error Metric
        if dimensions is None or not dimensions:
            dimensions = []
        if not isinstance(dimensions, list):
            raise merrors.MetricsCacheValueError(
                "Please provide a list of Dimension objects."
            )

        # don't add duplicate Dimension objects
        pre_dims_str = [str(d) for d in dimensions]

        if req_id is None:
            if not self._check_matching_dims("Level", pre_dims_str):
                dimensions.append(Dimension("Level", "Error"))
        else:
            if not self._check_matching_dims("ModelName", pre_dims_str):
                dimensions.append(Dimension("ModelName", self.model_name))
            if not self._check_matching_dims("Level", pre_dims_str):
                dimensions.append(Dimension("Level", "host"))

        dims_str = ",".join([str(d) for d in dimensions])
        metric_key = f"[{metric_type.value}]-[{name}]-[{dims_str}]"

        if metric_key not in self.cache:
            metric = Metric(
                name=name,
                value=value,
                unit=unit,
                dimensions=dimensions,
                request_id=req_id,
                metric_type=metric_type.value,
            )
            self.cache[metric_key] = metric
            self.store.append(metric)

        else:
            existing_metric = self.get_metric(
                metric_type=metric_type, metric_name=name, dimensions=dims_str
            )
            existing_metric.update(value)

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

    def add_counter(
        self,
        name: str,
        value: int or float,
        idx=None,
        dimensions: list = None,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ):
        """
        Add a counter metric or increment an existing counter metric
            Default metric type is counter
        Parameters
        ----------
        name : str
            metric name
        value: int or float
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to counter metric type for Counter metrics
        """
        unit = "count"
        req_id = self._get_req(idx)
        self._add_or_update(
            name=name,
            value=value,
            req_id=req_id,
            unit=unit,
            metric_type=metric_type,
            dimensions=dimensions,
        )

    def add_time(
        self,
        name: str,
        value: int or float,
        idx=None,
        unit: str = "ms",
        dimensions: list = None,
        metric_type: MetricTypes = MetricTypes.GAUGE,
    ):
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
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Time metrics
        """
        if unit not in ["ms", "s"]:
            raise merrors.MetricsCacheValueError(
                "the unit for a timed metric should be one of ['ms', 's']"
            )
        req_id = self._get_req(idx)
        self._add_or_update(
            name=name,
            value=value,
            req_id=req_id,
            unit=unit,
            metric_type=metric_type,
            dimensions=dimensions,
        )

    def add_size(
        self,
        name: str,
        value: int or float,
        idx=None,
        unit: str = "MB",
        dimensions: list = None,
        metric_type: MetricTypes = MetricTypes.GAUGE,
    ):
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
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Size metrics
        """
        if unit not in ["MB", "kB", "GB", "B"]:
            raise ValueError(
                "The unit for size based metric is one of ['MB','kB', 'GB', 'B']"
            )
        req_id = self._get_req(idx)
        self._add_or_update(
            name=name,
            value=value,
            req_id=req_id,
            unit=unit,
            metric_type=metric_type,
            dimensions=dimensions,
        )

    def add_percent(
        self,
        name: str,
        value: int or float,
        idx=None,
        dimensions: list = None,
        metric_type: MetricTypes = MetricTypes.GAUGE,
    ):
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
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Percent metrics
        """
        unit = "percent"
        req_id = self._get_req(idx)
        self._add_or_update(
            name=name,
            value=value,
            req_id=req_id,
            unit=unit,
            metric_type=metric_type,
            dimensions=dimensions,
        )

    def add_error(
        self,
        name: str,
        value: int or float,
        dimensions: list = None,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ):
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
        metric_type: MetricTypes
           type for defining different operations, defaulted to counter metric type for Error metrics
        """
        unit = ""
        # noinspection PyTypeChecker
        self._add_or_update(
            name=name,
            value=value,
            req_id=None,
            unit=unit,
            metric_type=metric_type,
            dimensions=dimensions,
        )

    def get_metric(
        self, metric_type: MetricTypes, metric_name: str, dimensions: list or str
    ) -> Metric:
        """
        Get a Metric from cache.
            Ask user for required requirements to form metric key to retrieve Metric.

        Parameters
        ----------
        metric_type: MetricTypes
            Type of metric: use MetricTypes enum to specify

        metric_name: str
            Name of metric

        dimensions: list or str
            list of dimension keys which should be strings

        """
        metric_key = f"[{metric_type.value}]-[{metric_name}]-[{dimensions}]"

        metric_obj = self.cache.get(metric_key)
        if metric_obj:
            logging.debug(f"Successfully received metric {metric_key}")
            return metric_obj
        else:
            raise merrors.MetricsCacheKeyError(
                f"Metric key {metric_key} does not exist."
            )

    def add_metric(
        self,
        metric_name: str,
        value: int or float,
        unit: str,
        idx=None,
        dimensions: list = None,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> None:
        """
        Create a new metric and add into cache.
            Add a metric which is generic with custom metrics

        Parameters
        ----------
        metric_name: str
            Name of metric
        value: int, float
            value of metric
        unit: str
            unit of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        metric_type: MetricTypes
            Type of metric
        """
        req_id = self._get_req(idx)
        if (
            not isinstance(metric_name, str)
            or not isinstance(unit, str)
            or not isinstance(metric_type, MetricTypes)
            or not isinstance(value, (float, int))
        ):
            raise merrors.MetricsCacheTypeError(
                f"metric_name must be a str, unit must be a str, "
                f"dimensions should be a list of Dimension objects/None, "
                f"metric type must be a MetricTypes enum, value must be a int/float"
            )

        # dimensions either has to be a list of Dimensions or empty list / None
        if (
            dimensions
            and not isinstance(dimensions, list)
            and not isinstance(dimensions[0], Dimension)
        ):
            raise merrors.MetricsCacheTypeError(
                f"Dimensions list is expected to be made up of Dimension objects."
            )

        # Make sure that the passed arguments follow a valid naming convention (doesn't contain certain characters)
        self._inspect_naming_convention(
            metric_name, unit, dimensions, metric_type, value
        )

        self._add_or_update(
            name=metric_name,
            value=value,
            req_id=req_id,
            unit=unit,
            metric_type=metric_type,
            dimensions=dimensions,
        )

        logging.info(f"Successfully added {metric_name} Metric object to cache.")

    @staticmethod
    def _inspect_naming_convention(*metric_arg) -> None:
        """
        Inspect naming convention for each argument being used to create a Metric object.

            Checking to ensure that certain symbols (- / []) are not used in args so that Metric strings
            can be created without ambiguity.

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
                    logging.warning(
                        f"There is a '{delim}' symbol found in {arg} argument. "
                        f"Please refrain from using the "
                        f"'{delim}' as it is used as the delimiter in the Metric object string."
                    )

        delimiters = [
            "-",
            "[",
            "]",
        ]  # list of symbols that should not be included in any Metric arguments

        for individual_metric_arg in metric_arg:
            if isinstance(individual_metric_arg, list):  # list of Dimension objects
                for dimension in individual_metric_arg:
                    _check_individual_arg(delimiters, dimension.__str__())
            else:  # should always be every other Metric argument (and should all be string parsable)
                _check_individual_arg(delimiters, individual_metric_arg)
