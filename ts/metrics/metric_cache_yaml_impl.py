"""
Metrics Cache class for creating objects from yaml spec
"""
import logging

import yaml

import ts.metrics.metric_cache_errors as merrors
from ts.metrics.caching_metric import CachingMetric
from ts.metrics.metric_cache_abstract import MetricCacheAbstract
from ts.metrics.metric_type_enum import MetricTypes

logger = logging.getLogger(__name__)


class MetricsCacheYamlImpl(MetricCacheAbstract):
    def __init__(self, config_file_path):
        """
        Constructor
            Passes yaml file and creates metrics objects.

        Parameters
        ----------
        config_file_path: str
            Path of yaml file to be parsed
        """
        super().__init__(config_file_path=config_file_path)
        self._parse_yaml_file(self.config_file_path)

    def _parse_yaml_file(self, config_file_path) -> None:
        """
        Parse yaml file using PyYAML library.
        """
        if not config_file_path:
            raise merrors.MetricsCacheTypeError("Config file not initialized")

        try:
            self._parsed_file = yaml.safe_load(
                open(config_file_path, "r", encoding="utf-8")
            )
            logging.info(f"Successfully loaded {config_file_path}.")
        except yaml.YAMLError as exc:
            raise merrors.MetricsCachePyYamlError(
                f"Error parsing file {config_file_path}: {exc}"
            )
        except IOError as io_err:
            raise merrors.MetricsCacheIOError(
                f"Error reading file {config_file_path}: {io_err}"
            )
        except Exception as err:
            raise merrors.GeneralMetricsCacheError(
                f"General error found in file {config_file_path}: {err}"
            )

    def _parse_metrics_section(self, key="model_metrics") -> dict:
        """
        Given a key present in the yaml, returns the corresponding section

        Parameters
        ----------
        key: str
            section of yaml file to be parsed
        """
        try:
            val = self._parsed_file[key]
        except KeyError as err:
            raise merrors.MetricsCacheKeyError(
                f"'{key}' key not found in yaml file: {err}"
            )
        logging.debug(f"Successfully parsed {key} section of yaml file")
        return val

    def initialize_cache(self) -> None:
        """
        Create Metric objects based off of the model_metrics data and add to cache
        """
        metrics_section = self._parse_metrics_section("model_metrics")
        if not metrics_section:
            raise merrors.MetricsCacheValueError(
                "Missing `model_metrics` specification"
            )
        for metric_type, metrics_list in metrics_section.items():
            try:
                metric_enum = MetricTypes(metric_type)
            except Exception as exc:
                raise merrors.MetricsCacheKeyError(f"Invalid metric type: {exc}")

            for metric in metrics_list:
                try:
                    metric_name = metric["name"]
                    unit = metric["unit"]
                    dimension_names = metric["dimensions"]
                    self.add_metric_to_cache(
                        metric_name=metric_name,
                        unit=unit,
                        dimension_names=dimension_names,
                        metric_type=metric_enum,
                    )
                except KeyError as k_err:
                    raise merrors.MetricsCacheKeyError(
                        f"Key not found in cache spec: {k_err}"
                    )

    def add_metric_to_cache(
        self,
        metric_name: str,
        unit: str,
        dimension_names: list = [],
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> CachingMetric:
        """
        Create a new metric and add into cache. Override existing metric with same name if present.

        Parameters
        ----------
        metric_name str
            Name of metric
        unit str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimension_names list
            list of dimension name strings for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        Returns
        -------
        newly created Metrics object
        """
        self._check_type(metric_name, str, "`metric_name` must be a str")
        self._check_type(unit, str, "`unit` must be a str")
        self._check_type(
            metric_type, MetricTypes, "`metric_type` must be a MetricTypes enum"
        )
        if dimension_names:
            self._check_type(
                dimension_names,
                list,
                "`dimension_names` should be a list of dimension name strings",
            )
        if metric_type not in self.cache.keys():
            self.cache[metric_type] = {}
        metric = CachingMetric(
            metric_name=metric_name,
            unit=unit,
            dimension_names=dimension_names,
            metric_type=metric_type,
        )
        if metric_name in self.cache[metric_type].keys():
            logging.warning(f"Overriding existing key {metric_type}:{metric_name}")
        self.cache[metric_type][metric_name] = metric
        return metric

    def get_metric(
        self,
        metric_name: str,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> CachingMetric:
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
        self._check_type(metric_name, str, "`metric_name` must be a str")
        self._check_type(
            metric_type, MetricTypes, "`metric_type` must be a MetricTypes enum"
        )
        try:
            metric = self.cache[metric_type][metric_name]
        except KeyError:
            raise merrors.MetricsCacheKeyError(
                f"Metric of type '{metric_type}' and name '{metric_name}' doesn't exist"
            )
        else:
            return metric

    def cache_keys(self):
        """
        Testing util method
        """
        keys = []
        for metric_type, metric in self.cache.items():
            for metric_name in metric.keys():
                keys.append(f"{metric_type.value}:{metric_name}")
        return keys
