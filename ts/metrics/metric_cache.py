"""
1. Parse model_metrics portion of yaml file and get the data
2. Create Metric objects based off of the model_metrics yaml data
3. Create hash table with key being metricType_metricName_dimensions
and value being the respective Metric object created in the previous step
"""
import sys
import yaml
import logging

from metric import Metric


class MetricsCaching:
    def __init__(self, yaml_file):
        """
        Constructor for MetricsCaching class

            MetricsCaching class will hold Metric objects in a cache that can be edited and read from.
            The class also has the ability to parse a yaml file into Metrics objects

        Parameters
        ----------
        yaml_file: str
            Name of yaml file to be parsed

        """
        self.backend_cache = {}  # hash table to store the Metric objects
        self.yaml_file = yaml_file

    def add_metric(self, metric_name: str, unit: str, dimensions: list, metric_type: str) -> None:
        """
        Create a new metric and add into cache

        Parameters
        ----------
        metric_name: str
            Name of metric
        unit: str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimensions: list
            list of dimension objects
        metric_type: str

        """

        print(f"Adding metric with fields of: metric name - {metric_name}, unit - {unit}, dimensions - {dimensions}, "
              f"metric type - {metric_type}")

        dims_str = "__".join([str(d) for d in dimensions])

        self.backend_cache[f"{metric_type}_{metric_name}_{dims_str}"] = Metric(name=metric_name,
                                                                               value=0,
                                                                               unit=unit,
                                                                               dimensions=dimensions)
        print("Successfully added metric.")

    def get_metric(self, metric_key: str) -> Metric:
        """
        Get a metric from cache

        Parameters
        ----------
        metric_key: str
            Key to identify a Metric object within the cache

        """
        print(f"Getting metric {metric_key}")
        metric_obj = self.backend_cache.get(metric_key)
        if metric_obj:
            print("Successfully received metric")
            return metric_obj
        else:
            print("Metric does not exist.")
            sys.exit(1)

    def _parse_yaml_file(self) -> dict:
        """
        Parse yaml file using PyYAML library.
        """
        yml_dict = None
        stream = open(self.yaml_file, "r", encoding="utf-8")
        try:
            yml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:  # TODO find more errors in UT
            logging.error(exc)
            sys.exit(1)
        except Exception as err:
            logging.warning(f"Bad config file - {err}")

        return yml_dict

    def _parse_specific_metric(self, yaml_section="model_metrics") -> dict:
        """
        Returns model_metrics portion of yaml file in a dict

        Parameters
        ----------
        yaml_section: str
            section of yaml file to be parsed

        """
        print(f"Parsing {yaml_section} section of yaml file...")
        yaml_hash_table = self._parse_yaml_file()
        try:
            model_metrics_table = yaml_hash_table[yaml_section]
        except Exception as err:  # TODO find more errors in UT. Find specific key error for this line
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
                except Exception as err:  # TODO find more errors in UT
                    print(f"{err}")
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
    backend_cache_obj = MetricsCaching("metrics_yaml_testing/metrics.yaml")
    backend_cache_obj.yaml_to_cache()

    # get metric method
    gauge_metric = backend_cache_obj.get_metric("gauge_None_model_name__host")
    print(gauge_metric.name)

    # add metric method
    backend_cache_obj.add_metric(metric_name="new", unit="ms", dimensions=["filler"], metric_type="type")

    new_metric = backend_cache_obj.get_metric("type_new_filler")
    print(new_metric.name)
