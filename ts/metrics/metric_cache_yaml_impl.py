"""
Metrics Cache class for creating objects from yaml spec
"""
import logging
import os
import yaml

import ts.metrics.metric_cache_errors as merrors
from ts.metrics.caching_metric import CachingMetric
from ts.metrics.metric_cache_abstract import MetricCacheAbstract
from ts.metrics.metric_type_enum import MetricTypes


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
        try:
            os.path.exists(config_file_path)
        except Exception as exc:
            raise merrors.MetricsCacheTypeError(f"Error loading {config_file_path} file: {exc}")

        super().__init__(request_ids=None, model_name=None, config_file_path=config_file_path)
        self._parse_yaml_file()

    def _parse_yaml_file(self) -> None:
        """
        Parse yaml file using PyYAML library.
        """
        if not self.config_file_path:
            raise merrors.MetricsCacheTypeError("Config file not initialized")

        try:
            self._parsed_file = yaml.safe_load(
                open(self.config_file_path, "r", encoding="utf-8"))
            logging.info(f"Successfully loaded {self.config_file_path}.")
        except yaml.YAMLError as exc:
            raise merrors.MetricsCachePyYamlError(
                f"Error parsing file {self.config_file_path}: {exc}"
            )
        except IOError as io_err:
            raise merrors.MetricsCacheIOError(
                f"Error reading file {self.config_file_path}: {io_err}"
            )
        except Exception as err:
            raise merrors.GeneralMetricsCacheError(
                f"General error found in file {self.config_file_path}: {err}"
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
        metrics_section = self._parse_metrics_section()
        for metric_type, metrics_list in metrics_section.items():
            try:
                metric_enum = MetricTypes(metric_type)
            except Exception as exc:
                raise merrors.MetricsCacheKeyError(f"Invalid metric type: {exc}")

            for metric in metrics_list:
                for metric_name, metric_val in metric:
                    unit = metric_val["unit"]
                    dimension_names = metric_val["dimensions"]
                    self.add_metric(
                        metric_name=metric_name,
                        unit=unit,
                        dimension_names=dimension_names,
                        metric_type=metric_enum,
                    )

    def add_metric(
        self,
        metric_name: str,
        unit: str,
        dimension_names: list = None,
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> CachingMetric:
        """
        Create a new metric and add into cache

        Parameters
        ----------
        metric_name str
            Name of metric

        unit str
            unit can be one of ms, percent, count, MB, GB or a generic string

        dimension_names list
            list of dimension keys which should be strings, or the complete log of dimensions

        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram

        Returns
        -------
        newly created Metrics object
        """
        if (
            not isinstance(metric_name, str)
            or not isinstance(unit, str)
            or not isinstance(metric_type, MetricTypes)
        ):
            raise merrors.MetricsCacheTypeError(
                f"metric_name must be a str, unit must be a str, "
                f"dimensions should be a list of Dimension objects/None, "
                f"metric type must be a MetricTypes enum"
            )
        if (
            dimension_names
            and not isinstance(dimension_names, list)
        ):
            raise merrors.MetricsCacheTypeError(
                f"`dimension_names` should be a list of dimension name strings."
            )
        if metric_type not in self.cache.keys():
            self.cache[metric_type] = dict()
        metric = CachingMetric(
            metric_name=metric_name,
            unit=unit,
            dimension_names=dimension_names,
            metric_type=metric_type
        )
        self.cache[metric_type][metric.metric_name] = metric
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
        if not isinstance(metric_type, MetricTypes) or not isinstance(metric_name, str):
            raise merrors.MetricsCacheTypeError(
                f"metric_type must be MetricTypes enum, metric_name must be a str."
            )
        try:
            metric = self.cache[metric_type][metric_name]
        except KeyError:
            raise merrors.MetricsCacheKeyError(
                f"Metric of type '{metric_type}' and name '{metric_name}' doesn't exist"
            )
        else:
            return metric
