"""
Metrics Cache class for YAML file parsing.
"""
import logging
import os
import yaml

import ts.metrics.metric_cache_errors as merrors
from ts.metrics.dimension import Dimension
from ts.metrics.metric_cache_abstract import MetricCacheAbstract
from ts.metrics.metric_type_enum import MetricTypes


class MetricsCacheYaml(MetricCacheAbstract):
    def __init__(self, request_ids, model_name, config_file):
        """
        Constructor for MetricsCachingYaml class

            Passes yaml file to abstract class.

        Parameters
        ----------
        model_name: str
            Name of the model in use

        config_file: str
            Path of yaml file to be parsed

        """
        try:
            os.path.exists(config_file)
        except Exception as exc:
            raise merrors.MetricsCacheTypeError(f"File {config_file} does not exist: {exc}")

        super().__init__(request_ids=request_ids, model_name=model_name, config_file=config_file)
        self._parse_yaml_file()

    def _parse_yaml_file(self) -> None:
        """
        Parse yaml file using PyYAML library.
        """
        if not self.config_file:
            raise merrors.MetricsCacheTypeError("Config file not initialized")

        try:
            stream = open(self.config_file, "r", encoding="utf-8")
            self._parsed_file = yaml.safe_load(stream)
            logging.info(f"Successfully loaded {self.config_file}.")
        except yaml.YAMLError as exc:
            raise merrors.MetricsCachePyYamlError(
                f"Error parsing file {self.config_file}: {exc}"
            )
        except IOError as io_err:
            raise merrors.MetricsCacheIOError(
                f"Error reading file {self.config_file}: {io_err}"
            )
        except Exception as err:
            raise merrors.GeneralMetricsCacheError(
                f"General error found in file {self.config_file}: {err}"
            )

    def _parse_metrics_section(self, metrics_section="model_metrics") -> dict:
        """
        Returns specified portion of yaml file in a dict

        Parameters
        ----------
        metrics_section: str
            section of yaml file to be parsed

        """
        try:
            metrics_table = self._parsed_file[metrics_section]
        except KeyError as err:
            raise merrors.MetricsCacheKeyError(
                f"'{metrics_section}' key not found in yaml file: {err}"
            )
        logging.debug(f"Successfully parsed {metrics_section} section of yaml file")
        return metrics_table

    def _yaml_to_cache_util(self, metrics_section: dict) -> None:
        """
        Create Metric objects based off of the model_metrics yaml data and add to hash table

        Parameters
        ----------
        metrics_section: dict
            Parsed portion of the yaml file

        """
        if not isinstance(metrics_section, dict):
            raise merrors.MetricsCacheTypeError(
                f"{metrics_section} section is None and does not exist"
            )

        # get dimensions dictionary from yaml file
        dimensions_dict = self._parse_metrics_section("dimensions")
        logging.debug("Creating Metric objects")
        for metric_type, metric_attributes_list in metrics_section.items():
            try:  # get metric type as enum
                metric_type_enum = MetricTypes(metric_type)
            except Exception as exc:
                raise merrors.MetricsCacheKeyError(f"Metric type does not exist: {exc}")

            for (
                metric_type_dict
            ) in (
                metric_attributes_list
            ):  # dict of all metrics specific to one metric type
                for (
                    metric_name,
                    individual_metric_dict,
                ) in metric_type_dict.items():  # individual metric entries
                    try:
                        dimensions = []
                        unit = individual_metric_dict["unit"]
                        dimensions_list = individual_metric_dict["dimensions"]

                        # Create dimensions objects and add to list to be passed to add_metric
                        if dimensions_list:
                            for k, v in dimensions_dict.items():
                                if v in dimensions_list:
                                    dimensions.append(Dimension(k, v))

                        self.add_metric(
                            metric_name=metric_name,
                            value=0,
                            unit=unit,
                            dimensions=dimensions,
                            metric_type=metric_type_enum,
                        )

                    except KeyError as k_err:
                        raise merrors.MetricsCacheKeyError(f"Key not found: {k_err}")
                    except TypeError as t_err:
                        raise merrors.MetricsCacheTypeError(f"{t_err}")

    def parse_yaml_to_cache(self) -> None:
        """
        Parses specific portion of yaml file and creates Metrics objects and adds to the cache
        """
        metrics_section = self._parse_metrics_section()
        self._yaml_to_cache_util(metrics_section=metrics_section)

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
        Create a new metric and add into cache

            Using dimensions that are based on the yaml file specified

        Parameters
        ----------
        metric_name: str
            Name of metric
        unit: str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimensions: list
            list of dimension objects/strings read from yaml file
        metric_type: MetricTypes
            Type of metric
        idx: str
            Request id which is currently a UUID
        value: int, float
            value of metric
        """
        if dimensions and not isinstance(dimensions, list):
            raise merrors.MetricsCacheTypeError(
                "Dimensions has to be a list of string "
                "(which will be converted to list of Dimensions)"
                "/list of Dimension objects and cannot be empty/None"
            )

        # if already list of Dimension objects/None, then do nothing
        if not dimensions or isinstance(dimensions[0], Dimension):
            pass
        # if dimensions are list of strings, convert them to Dimension objects based on yaml file.
        elif isinstance(dimensions[0], str):
            dimensions_list = dimensions
            dimensions = []
            # get dimensions dictionary from yaml file
            dimensions_dict = self._parse_metrics_section("dimensions")
            try:
                # Create dimensions objects and add to list to be passed to add_metric
                for k, v in dimensions_dict.items():
                    if k in dimensions_list:
                        dimensions.append(Dimension(k, v))
            except Exception as err:
                raise merrors.MetricsCacheKeyError(f"Dimension not found: {err}")
        else:
            raise merrors.MetricsCacheTypeError(
                f"Dimensions have to either be a list of strings "
                f"(which will be converted to list of Dimension objects)"
                f" or a list of Dimension objects."
            )

        super().add_metric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            idx=idx,
            dimensions=dimensions,
            metric_type=metric_type,
        )

    def get_metric(
        self, metric_type: MetricTypes, metric_name: str, dimensions: list or str
    ):
        """
        Get a Metric from cache.
            Ask user for required requirements to form metric key to retrieve Metric.

        Parameters
        ----------
        metric_type: str
            Type of metric: use MetricTypes enum to specify

        metric_name: str
            Name of metric

        dimensions: list or str
            list of dimension keys which should be strings, or the complete log of dimensions

        """
        dims_str = None
        if not isinstance(metric_type, MetricTypes) or not isinstance(metric_name, str):
            raise merrors.MetricsCacheTypeError(
                f"metric_type must be MetricTypes enum, metric_name must be a str."
            )

        if isinstance(dimensions, str):
            dims_str = dimensions
        elif isinstance(dimensions, list):
            dimensions_dict = self._parse_metrics_section("dimensions")
            complete_dimensions = []
            for k, v in dimensions_dict.items():
                if k in dimensions:
                    complete_dimensions.append(Dimension(k, v))
            dims_str = ",".join([str(d) for d in complete_dimensions])
        else:
            merrors.MetricsCacheTypeError(
                f"{dimensions} is expected to be a string (complete Dimensions log line) "
                f"or list of strings (list of Dimension keys)"
            )
        return super().get_metric(metric_type, metric_name, dims_str)
