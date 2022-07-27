"""
Unit testing for MetricsCacheYaml class and yaml parsing.
"""
import sys

import pytest
from ts.metrics.dimension import Dimension
from ts.metrics.metric_cache_yaml import MetricsCacheYaml


class TestAddMetrics:

    def test_add_metric_passing(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type="type")

        assert "type-new_metric-model_name:example_model_name-host:example_host_name" in metrics_cache_obj.cache

    def test_add_metric_duplicate_passing(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type="type")

        assert "type-new_metric-model_name:example_model_name-host:example_host_name" in metrics_cache_obj.cache

        metric = metrics_cache_obj.get_metric("type-new_metric-model_name:example_"
                                              "model_name-host:example_host_name")
        assert metric.value == 0

        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type="type",
                                     value=42.5)

        metric = metrics_cache_obj.get_metric("type-new_metric-model_name:example_"
                                              "model_name-host:example_host_name")
        assert metric.value == 42.5

    def test_add_metric_dimensions_correct_obj_pass(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=[Dimension("hello", "world")],
                                     metric_type="type")

        for key, metric in metrics_cache_obj.cache.items():

            for dimensions_list in metric.dimensions:
                assert isinstance(dimensions_list, Dimension)

    def test_add_metric_dimensions_correct_obj_fail(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")

        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="new_metric",
                                         unit="ms",
                                         dimensions=Dimension("hello", "world"),
                                         metric_type="type")
        assert str(exc_info.value) == "Dimensions has to be a list of string " \
                                      "(which will be converted to list of Dimensions)/list of " \
                                      "Dimension objects and cannot be empty/None"

    def test_add_metric_fail_metric_name(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name=42,
                                         unit="ms",
                                         dimensions=["model_name", "host"],
                                         metric_type="type")
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions must be a list of " \
                                      "Dimension objects, metric type must be a str, value must be a int/float"

    def test_add_metric_fail_unit(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit=["foo"],
                                         dimensions=["model_name", "host"],
                                         metric_type="type")
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions must be a list of " \
                                      "Dimension objects, metric type must be a str, value must be a int/float"

    def test_add_metric_fail_dimensions(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit="ms",
                                         dimensions="sink",
                                         metric_type="type")
        assert str(exc_info.value) == "Dimensions has to be a list of string " \
                                      "(which will be converted to list of Dimensions)/list of Dimension" \
                                      " objects and cannot be empty/None"

    def test_add_metric_fail_dimensions_none(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit="ms",
                                         dimensions=None,
                                         metric_type="type")
        assert str(exc_info.value) == 'Dimensions has to be a list of string ' \
                                      '(which will be converted to list of Dimensions)/list ' \
                                      'of Dimension objects and cannot be empty/None'

    def test_add_metric_fail_type(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit="ms",
                                         dimensions=["model_name", "host"],
                                         metric_type={"key": 42})
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions must be a list of " \
                                      "Dimension objects, metric type must be a str, value must be a int/float"

    def test_add_metric_fail_type_unit(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(TypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit=["ms"],
                                         dimensions=["model_name", "host"],
                                         metric_type={"key": 42})
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions must be a list of " \
                                      "Dimension objects, metric type must be a str, value must be a int/float"


class TestGetMetrics:

    def test_get_metric_passing(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type="type")

        temp_metrics = metrics_cache_obj.get_metric("type-new_metric-model_name:"
                                                    "example_model_name-host:example_host_name")
        assert "new_metric" == temp_metrics.name
        assert isinstance(temp_metrics.dimensions, list)
        assert "Milliseconds" == temp_metrics.unit
        assert 0 == temp_metrics.value

    def test_get_metric_fail_not_exist(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type="type")

        with pytest.raises(SystemExit):
            metrics_cache_obj.get_metric("type-new_metric-dim1:dim3")

    def test_get_metric_fail_invalid_type(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj.get_metric(["type-new_metric-dim1:dim3"])


class TestParseYaml:

    def test_parse_expected_yaml(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        actual_dict = metrics_cache_obj._parse_yaml_file()
        expected_dict = {'dimensions': {'host': 'example_host_name',
                                        'model_name': 'example_model_name'},
                         'model_metrics': {'counter': [{'InferenceTimeInMS': {'dimensions': ['model_name',
                                                                                             'host'],
                                                                              'unit': 'ms'}},
                                                       {'NumberOfMetrics': {'dimensions': ['model_name'],
                                                                            'unit': 'count'}}],
                                           'gauge': [{'GaugeModelMetricNameExample': {'dimensions': ['model_name',
                                                                                                     'host'],
                                                                                      'unit': 'ms'}}],
                                           'histogram': [
                                               {'HistogramModelMetricNameExample': {'dimensions': ['model_name',
                                                                                                   'host'],
                                                                                    'unit': 'ms'}}]},
                         'ts_metrics': {'counter': [{'name': {'dimensions': ['model_name', 'host'],
                                                              'unit': 'ms'}}],
                                        'gauge': [{'name': {'dimensions': ['model_name', 'host'],
                                                            'unit': 'ms'}}],
                                        'histogram': [{'name': {'dimensions': ['model_name', 'host'],
                                                                'unit': 'ms'}}]}}
        assert actual_dict == expected_dict

    def test_yaml_file_passing(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")

        assert "metrics.yaml" == metrics_cache_obj.file

    def test_yaml_file_none_fail(self):
        with pytest.raises(TypeError) as exc_info:
            MetricsCacheYaml(None)
        assert str(exc_info.value) == "File passed must be a valid string path that exists."

    def test_yaml_file_non_yaml_extension_fail(self):
        with pytest.raises(TypeError) as exc_info:
            MetricsCacheYaml("metric_cache_unit_test.py")
        assert str(exc_info.value) == "Inputted file does not have a valid yaml file extension."

    def test_yaml_file_non_exist_fail(self):
        with pytest.raises(TypeError) as exc_info:
            MetricsCacheYaml("doesnt_exist.yaml")
        assert str(exc_info.value) == "File passed must be a valid string path that exists."

    def test_parse_yaml_errors(self):
        metrics_cache_obj = MetricsCacheYaml("metric_errors.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_yaml_file()


class TestParseModelMetrics:
    def test_pass_parse_model_metrics(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        assert isinstance(metrics_cache_obj._parse_specific_metric(), dict)

    def test_pass_parse_ts_metrics(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        assert isinstance(metrics_cache_obj._parse_specific_metric("ts_metrics"), dict)

    def test_fail_parse_model_metrics(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_wo_model_metrics.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_specific_metric()

    def test_fail_parse_none(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_wo_model_metrics.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_specific_metric(None)

    def test_fail_parse_missing_fields(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_model_empty.yaml")
        assert metrics_cache_obj._parse_specific_metric() is None


class TestYamlCacheUtil:
    def test_yaml_to_cache_util_pass(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        assert ['counter-InferenceTimeInMS-model_name:example_model_name-host:example_host_name',
                'counter-NumberOfMetrics-model_name:example_model_name',
                'gauge-GaugeModelMetricNameExample-model_name:example_model_name-host:example_host_name',
                'histogram-HistogramModelMetricNameExample-model_name:example_model_name-host:example_host_name'] \
               == list(metrics_cache_obj.cache.keys())

    def test_yaml_to_cache_util_fail_missing_fields(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_missing_types.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)

    def test_yaml_to_cache_util_fail_none(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_missing_types.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(None)

    def test_yaml_to_cache_util_fail_empty_section(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_model_empty.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)

    def test_yaml_to_cache_util_fail_empty_dimensions(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_empty_fields.yml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(TypeError):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)


class TestYamlCache:
    def test_yaml_to_cache_pass(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.parse_yaml_to_cache()
        assert ['counter-InferenceTimeInMS-model_name:example_model_name-host:example_host_name',
                'counter-NumberOfMetrics-model_name:example_model_name',
                'gauge-GaugeModelMetricNameExample-model_name:example_model_name-host:example_host_name',
                'histogram-HistogramModelMetricNameExample-model_name:example_model_name-host:example_host_name'] == list(
            metrics_cache_obj.cache.keys())


class TestManualAddMetricDimensions:
    def test_dimensions_metric_add_dimensions_custom_pass(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric("TempName", "count", [Dimension("MyDim", "MyDimValue")], "counter", 13.3)
        assert list(metrics_cache_obj.cache.keys()) == ["counter-TempName-MyDim:MyDimValue"]

    def test_dimensions_metric_add_dimensions_yaml_pass(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric("TempName", "count", ["model_name"], "counter", 13.3)
        assert list(metrics_cache_obj.cache.keys()) == ["counter-TempName-model_name:example_model_name"]

    def test_dimensions_metric_add_dimensions_and_yaml_pass(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        metrics_cache_obj.add_metric("TempName", "count", ["model_name"], "counter", 13.3)
        metrics_cache_obj.parse_yaml_to_cache()
        assert list(metrics_cache_obj.cache.keys()) == ['counter-TempName-model_name:example_model_name',
                                                        'counter-InferenceTimeInMS-model_name:'
                                                        'example_model_name-host:example_host_name',
                                                        'counter-NumberOfMetrics-model_name:example_model_name',
                                                        'gauge-GaugeModelMetricNameExample-model_name:example_'
                                                        'model_name-host:example_host_name',
                                                        'histogram-HistogramModelMetricNameExample-'
                                                        'model_name:example_model_name-host:example_host_name']

    def test_dimensions_metric_add_dimensions_yaml_nonexistent(self):
        metrics_cache_obj = MetricsCacheYaml("metrics.yaml")
        with pytest.raises(KeyError) as exc_info:
            metrics_cache_obj.add_metric("TempName", "count", ["not_existing"], "counter", 13.3)
        assert str(exc_info.value) == '"Dimension not found: \'not_existing\'"'

    def test_dimensions_metric_dimension_not_present(self):
        metrics_cache_obj = MetricsCacheYaml("metrics_mismatching_dims.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj.parse_yaml_to_cache()

    def test_dimensions_metric_dimension_warning_name_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml("metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("Temp-Name", "count", ["model_name"], "counter", 13.3)

        assert "WARNING  root:metric_cache_abstract.py:116 There is a '-' symbol found in Temp-Name argument. " \
               "Please refrain from using the '-' as it is used as the delimiter in the Metric object string.\n" \
               == caplog.text

    def test_dimensions_metric_dimension_warning_unit_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml("metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", "count-unit", ["model_name"], "counter", 13.3)

        assert "WARNING  root:metric_cache_abstract.py:116 There is a '-' symbol found in count-unit argument. " \
               "Please refrain from using the '-' as it is used as the delimiter in the Metric object string.\n" \
               == caplog.text

    def test_dimensions_metric_dimension_warning_dim_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml("metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", "count", [Dimension("hello-world", "foo")], "counter", 13.3)

        assert "WARNING  root:metric_cache_abstract.py:116 There is a '-' symbol found in hello-world:foo argument. " \
               "Please refrain from using the '-' as it is used as the delimiter in the Metric object string.\n" \
               == caplog.text

    def test_dimensions_metric_dimension_warning_dim_two_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml("metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", "count", [Dimension("helloworld", "foo-bar")], "counter", 13.3)

        assert "WARNING  root:metric_cache_abstract.py:116 There is a '-' symbol found in helloworld:foo-bar argument. " \
               "Please refrain from using the '-' as it is used as the delimiter in the Metric object string.\n" \
               == caplog.text

    def test_dimensions_metric_dimension_warning_type_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml("metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", "count", [Dimension("helloworld", "foobar")], "counter-type", 13.3)

        assert "WARNING  root:metric_cache_abstract.py:116 There is a '-' symbol found in counter-type argument. " \
               "Please refrain from using the '-' as it is used as the delimiter in the Metric object string.\n" \
               == caplog.text


if __name__ == '__main__':
    pytest.main()
