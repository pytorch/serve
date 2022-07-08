"""
1. Parse model_metrics portion of yaml file and get the data
2. Create Metric objects based off of the model_metrics yaml data
3. Create hash table with key being metricType_metricName_dimensions
and value being the respective Metric object created in the previous step

naming and unit testing
"""
import sys
import yaml
import psutil

from ts.metrics.metric import Metric
from ts.metrics.dimension import Dimension


class MetricsCache:
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
                    temp_dimensions.append(Dimension(name=dimensions[i], value=dimensions[i+1]))
            dimensions = temp_dimensions
        else:
            raise ValueError(f"Dimensions list is expected to be an even number if the list of dimensions"
                             f" is made up of strings.")

        print(f"Adding metric with fields of: metric name - {metric_name}, unit - {unit}, dimensions - {dimensions}, "
              f"metric type - {metric_type}")

        dims_str = "-".join([str(d) for d in dimensions])

        self.backend_cache[f"{metric_type}-{metric_name}-{dims_str}"] = Metric(name=metric_name,
                                                                               value=value,
                                                                               unit=unit,
                                                                               dimensions=dimensions,
                                                                               metric_type=metric_type)
        print("Successfully added metric.")

    def get_metric(self, metric_key: str) -> Metric:
        """
        Get a metric from cache

        Parameters
        ----------
        metric_key: str
            Key to identify a Metric object within the cache

        """
        if not isinstance(metric_key, str):
            print(f"Only string types are acceptable as argument.")
            sys.exit(1)

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
        if not self.yaml_file:
            print("No yaml file detected.")
            sys.exit(1)
        yml_dict = None
        try:
            stream = open(self.yaml_file, "r", encoding="utf-8")
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
    backend_cache_obj = MetricsCache("../tests/metrics_yaml_testing/metrics.yaml")
    backend_cache_obj.yaml_to_cache()

    # Adding 1 host metric (CPUUtil) and 1 model metric (# of inferences),
    # update the metric,
    # and add to MetricsCache
    dimension = [Dimension('Level', 'Host')]
    # FIXME should i also use the existing Dimension class? probably yes
    cpu_util_data = psutil.cpu_percent()
    backend_cache_obj.add_metric(metric_name="CPUUtilization", value=cpu_util_data, unit="percent",
                                 dimensions=dimension, metric_type="CPUUtilizationType")
    print("================")
    print(f"CPU UTIL METRIC")
    cpu_util_metric = backend_cache_obj.get_metric("CPUUtilizationType-CPUUtilization-Level:Host")
    print(cpu_util_metric)
    cpu_util_metric.update(2.48)
    print(cpu_util_metric)
