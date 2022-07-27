"""
Metrics Cache class for YAML file parsing.
"""
import sys
import yaml
import os
import logging
import argparse

from ts.metrics.metric_cache_abstract import MetricCacheAbstract
from ts.service import emit_metrics
from ts.metrics.dimension import Dimension


class MetricsCacheYaml(MetricCacheAbstract):
    def __init__(self, yaml_file):
        """
        Constructor for MetricsCachingYaml class

            Passes yaml file to abstract class.

        Parameters
        ----------
        yaml_file: str
            Name of yaml file to be parsed

        """
        yaml_file_extensions = [".yaml", ".yml"]
        extension_bool = False
        if not yaml_file or not isinstance(yaml_file, str) or not os.path.exists(yaml_file):
            raise TypeError(f"File passed must be a valid string path that exists.")

        for extension in yaml_file_extensions:
            if yaml_file.endswith(extension):
                extension_bool = True
                break

        if not extension_bool:
            raise TypeError(f"Inputted file does not have a valid yaml file extension.")

        super().__init__(yaml_file)

    def _parse_yaml_file(self) -> dict:
        """
        Parse yaml file using PyYAML library.
        """
        # TODO: look into standard doc convention for private methods in python
        if not self.file:
            logging.error("No yaml file detected.")
            sys.exit(1)
        yml_dict = None
        try:
            logging.debug(f"Reading in yaml file...")
            stream = open(self.file, "r", encoding="utf-8")
            yml_dict = yaml.safe_load(stream)
            logging.info(f"Successfully loaded yaml file.")
        except yaml.YAMLError as exc:
            logging.error(f"Error parsing file: {exc}")
            sys.exit(1)
        except IOError as io_err:
            logging.error(f"Error reading file: {io_err}")
            sys.exit(1)
        except Exception as err:
            logging.error(f"General error: {err}")
            sys.exit(1)

        return yml_dict

    def _parse_specific_metric(self, yaml_section="model_metrics") -> dict:
        """
        Returns specified portion of yaml file in a dict

        Parameters
        ----------
        yaml_section: str
            section of yaml file to be parsed

        """
        yaml_hash_table = self._parse_yaml_file()

        logging.debug(f"Parsing {yaml_section} section of yaml file...")
        try:
            metrics_table = yaml_hash_table[yaml_section]
        except KeyError as err:
            logging.error(f"'{yaml_section}' key not found in yaml file - {err}")
            sys.exit(1)
        logging.info(f"Successfully parsed {yaml_section} section of yaml file")
        return metrics_table

    def _yaml_to_cache_util(self, specific_metrics_table: dict) -> None:
        """
        Create Metric objects based off of the model_metrics yaml data and add to hash table

        Parameters
        ----------
        specific_metrics_table: dict
            Parsed portion of the yaml file

        """
        if not isinstance(specific_metrics_table, dict):
            logging.error(f"metrics section is None and does not exist")
            sys.exit(1)

        # get dimensions dictionary from yaml file
        dimensions_dict = self._parse_specific_metric("dimensions")

        logging.info("Creating Metric objects")
        for metric_type, metric_attributes_list in specific_metrics_table.items():
            metric_name = None
            unit = None
            dimensions = None

            for metric_type_dict in metric_attributes_list:  # dict of all metrics specific to one metric type
                for metric_name, individual_metric_dict in metric_type_dict.items():  # individual metric entries
                    try:
                        dimensions = []
                        metric_name = metric_name
                        unit = individual_metric_dict["unit"]
                        dimensions_list = individual_metric_dict["dimensions"]

                        # Create dimensions objects and add to list to be passed to add_metric
                        for dimension in dimensions_list:
                            dimensions.append(Dimension(dimension, dimensions_dict[dimension]))

                        self.add_metric(metric_name=metric_name,
                                        unit=unit,
                                        dimensions=dimensions,
                                        metric_type=metric_type
                                        )

                    except KeyError as err:
                        logging.error(f"Key not found: {err}")
                        sys.exit(1)
        logging.info(f"Successfully created Metric objects.")

    def parse_yaml_to_cache(self):
        """
        Parses specific portion of yaml file and creates Metrics objects and adds to the cache
        """
        specific_metrics_table = self._parse_specific_metric()
        self._yaml_to_cache_util(specific_metrics_table=specific_metrics_table)

    def add_metric(self, metric_name: str, unit: str, dimensions: list, metric_type: str, value=0) -> None:
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
        metric_type: str
            Type of metric
        value: int, float
            value of metric
        """

        if not isinstance(dimensions, list) or not dimensions:
            raise TypeError("Dimensions has to be a list of string (which will be converted to list of Dimensions)"
                            "/list of Dimension objects and cannot be empty/None")

        # if dimensions are list of strings, convert them to Dimension objects based on yaml file.
        if isinstance(dimensions[0], str):
            dimensions_list = dimensions
            dimensions = []

            try:
                # get dimensions dictionary from yaml file
                dimensions_dict = self._parse_specific_metric("dimensions")

                # Create dimensions objects and add to list to be passed to add_metric
                for dimension in dimensions_list:
                    dimensions.append(Dimension(dimension, dimensions_dict[dimension]))
            except Exception as err:
                raise KeyError(f"Dimension not found: {err}")

        elif isinstance(dimensions[0], Dimension):
            pass
        else:
            raise TypeError(f"Dimensions have to either be a list of strings "
                            f"(which will be converted to list of Dimension objects) or a list of Dimension objects.")

        super().add_metric(metric_name=metric_name,
                           unit=unit,
                           dimensions=dimensions,
                           metric_type=metric_type,
                           value=value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="YAML File to be parsed",
        type=str,
        required=True
    )
    arguments = parser.parse_args()

    backend_cache_obj = MetricsCacheYaml(arguments.file)
    backend_cache_obj.parse_yaml_to_cache()
    emit_metrics(list(backend_cache_obj.cache.values()))
