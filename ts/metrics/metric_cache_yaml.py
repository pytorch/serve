"""
Metrics Cache class for YAML file parsing.
"""
import sys
import yaml
import psutil
import os

from ts.metrics.dimension import Dimension
from ts.metrics.metric_cache_abstract import MetricCacheAbstract


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
        if not self.file:
            print("No yaml file detected.")
            sys.exit(1)
        yml_dict = None
        try:
            stream = open(self.file, "r", encoding="utf-8")
            yml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error parsing file: {exc}")
            sys.exit(1)
        except IOError as io_err:
            print(f"Error reading file: {io_err}")
            sys.exit(1)
        except Exception as err:
            print(f"General error: {err}")
            sys.exit(1)

        return yml_dict

    def _parse_specific_metric(self, yaml_section="model_metrics") -> dict:
        """
        Returns model_metrics portion of yaml file in a dict

        Parameters
        ----------
        yaml_section: str
            section of yaml file to be parsed

        """
        yaml_hash_table = self._parse_yaml_file()

        print(f"Parsing {yaml_section} section of yaml file...")
        try:
            model_metrics_table = yaml_hash_table[yaml_section]
        except KeyError as err:
            print(f"'{yaml_section}' key not found in yaml file - {err}")
            sys.exit(1)
        print(f"Successfully parsed {yaml_section} section of yaml file")
        return model_metrics_table

    def _yaml_to_cache_util(self, model_metrics_table: dict) -> None:
        """
        Create Metric objects based off of the model_metrics yaml data and add to hash table

        Parameters
        ----------
        model_metrics_table: dict
            Parsed portion of the yaml file

        """
        if not isinstance(model_metrics_table, dict):
            print(f"model metrics is None and does not exist")
            sys.exit(1)

        print("Creating Metric objects")
        for metric_type, metric_attributes_list in model_metrics_table.items():
            metric_name = None
            unit = None
            dimensions = None
            for metric_dict in metric_attributes_list:
                try:
                    metric_name = metric_dict["name"]
                    unit = metric_dict["unit"]
                    dimensions = metric_dict["dimensions"]
                except KeyError as err:
                    print(f"Key not found: {err}")
                    sys.exit(1)

            self.add_metric(metric_name=metric_name, unit=unit, dimensions=dimensions, metric_type=metric_type)

        print("Completed creating Metric objects")

    def yaml_to_cache(self):
        """
        Parses specific portion of yaml file and creates Metrics objects and adds to the cache
        """
        model_metrics_table = self._parse_specific_metric()
        self._yaml_to_cache_util(model_metrics_table=model_metrics_table)


if __name__ == "__main__":
    # YAML to cache
    backend_cache_obj = MetricsCacheYaml("../tests/metrics_yaml_testing/metrics.yaml")
    backend_cache_obj.yaml_to_cache()

    # Adding 1 host metric (CPUUtil) and 1 model metric (# of inferences),
    # update the metric,
    # and add to MetricsCache
    dimension = [Dimension('Level', 'Host')]

    cpu_util_data = psutil.cpu_percent()
    backend_cache_obj.add_metric(metric_name="CPUUtilization", value=cpu_util_data, unit="percent",
                                 dimensions=dimension, metric_type="CPUUtilizationType")
    print("================")
    print(f"CPU UTIL METRIC")
    cpu_util_metric = backend_cache_obj.get_metric("CPUUtilizationType-CPUUtilization-Level:Host")
    print(cpu_util_metric)
    cpu_util_metric.update(2.48)
    print(cpu_util_metric)
