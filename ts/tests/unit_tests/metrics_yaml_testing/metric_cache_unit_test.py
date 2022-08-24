"""
Unit testing for ts/metrics/metric_cache_abstract.py,
ts/metrics/metric_cache_yaml.py, and emit_metrics() ts/service.py
"""
import pytest
import uuid
import ts.metrics.metric_cache_errors as merrors

from ts.metrics.dimension import Dimension
from ts.metrics.metric_cache_yaml import MetricsCacheYaml
from ts.metrics.metrics_store import MetricsStore
from ts.metrics.metric_type_enums import MetricTypes
from ts.metrics.metric import Metric
from ts.service import emit_metrics


class TestAddMetrics:

    def test_add_metric_passing(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["ModelName", "host"],
                                     metric_type=MetricTypes.gauge)

        assert "[gauge]-[new_metric]-[ModelName:my_tc,host:example_host_name,Level:Model]" \
               in metrics_cache_obj.cache

    def test_add_metric_duplicate_passing(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["ModelName"],
                                     metric_type=MetricTypes.gauge)

        assert "[gauge]-[new_metric]-[ModelName:my_tc,Level:Model]" in metrics_cache_obj.cache

        metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                              "ModelName:my_tc,Level:Model")
        assert metric.value == 123.5

        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["ModelName"],
                                     metric_type=MetricTypes.gauge,
                                     value=42.5)

        metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                              "ModelName:my_tc,Level:Model")
        assert metric.value == 42.5

    def test_add_metric_dimensions_correct_obj_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=[Dimension("hello", "world")],
                                     metric_type=MetricTypes.gauge)

        for key, metric in metrics_cache_obj.cache.items():
            for dimensions_list in metric.dimensions:
                assert isinstance(dimensions_list, Dimension)

    def test_add_metric_dimensions_correct_obj_fail(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")

        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="new_metric",
                                         value=123.5,
                                         unit="ms",
                                         dimensions=Dimension("hello", "world"),
                                         metric_type=MetricTypes.gauge)
        assert str(exc_info.value) == "Dimensions has to be a list of string " \
                                      "(which will be converted to list of Dimensions)/list of " \
                                      "Dimension objects and cannot be empty/None"

    def test_add_metric_fail_metric_name(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name=42,
                                         value=123.5,
                                         unit="ms",
                                         dimensions=["model_name", "host"],
                                         metric_type=MetricTypes.gauge)
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions should be a list of " \
                                      "Dimension objects/None, metric type must be a MetricTypes enum, " \
                                      "value must be a int/float"

    def test_add_metric_fail_unit(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         value=123.5,
                                         unit=["foo"],
                                         dimensions=["model_name", "host"],
                                         metric_type=MetricTypes.gauge)
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions should be a list of " \
                                      "Dimension objects/None, metric type must be a MetricTypes enum, " \
                                      "value must be a int/float"

    def test_add_metric_fail_dimensions(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         value=123.5,
                                         unit="ms",
                                         dimensions="sink",
                                         metric_type=MetricTypes.gauge)
        assert str(exc_info.value) == "Dimensions has to be a list of string " \
                                      "(which will be converted to list of Dimensions)/list of Dimension" \
                                      " objects and cannot be empty/None"

    def test_add_metric_fail_type(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         value=123.5,
                                         unit="ms",
                                         dimensions=["model_name", "host"],
                                         metric_type={"key": 42})
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions should be a list of " \
                                      "Dimension objects/None, metric type must be a MetricTypes enum, " \
                                      "value must be a int/float"

    def test_add_metric_fail_type_unit(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(metric_name="bar",
                                         value=123.5,
                                         unit=["ms"],
                                         dimensions=["model_name", "host"],
                                         metric_type={"key": 42})
        assert str(exc_info.value) == "metric_name must be a str, " \
                                      "unit must be a str, " \
                                      "dimensions should be a list of " \
                                      "Dimension objects/None, metric type must be a MetricTypes enum, " \
                                      "value must be a int/float"


class TestGetMetrics:

    def test_get_metric_passing(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type=MetricTypes.gauge)

        temp_metrics = metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                                    "model_name:example_model_name,host:example_host_name,"
                                                    "ModelName:ModelNameExample,Level:Model")
        assert temp_metrics.name == "new_metric"
        assert isinstance(temp_metrics.dimensions, list)
        assert temp_metrics.unit == "Milliseconds"
        assert temp_metrics.value == 123.5

    def test_get_metric_dimensions_list_passing(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["host", "ModelName", "Level"],
                                     metric_type=MetricTypes.gauge)

        temp_metrics = metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                                    ["host", "ModelName", "Level"])
        assert temp_metrics.name == "new_metric"
        assert isinstance(temp_metrics.dimensions, list)
        assert temp_metrics.unit == "Milliseconds"
        assert temp_metrics.value == 123.5

    def test_get_metric_dimensions_list_nonexist(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["host", "ModelName", "Level"],
                                     metric_type=MetricTypes.gauge)

        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                         ["host", "ModelNames", "Level"])

        assert str(exc_info.value) == "'Metric key [gauge]-[new_metric]-[host:example_host_name,Level:Model] " \
                                      "does not exist.'"

    def test_get_metric_dimensions_list_none(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["host", "ModelName", "Level"],
                                     metric_type=MetricTypes.gauge)

        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                         None)

        assert str(exc_info.value) == "'Metric key [gauge]-[new_metric]-[None] " \
                                      "does not exist.'"

    def test_get_metric_dimensions_list_empty(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["host", "ModelName", "Level"],
                                     metric_type=MetricTypes.gauge)

        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric",
                                         [])

        assert str(exc_info.value) == "'Metric key [gauge]-[new_metric]-[] " \
                                      "does not exist.'"

    def test_get_metric_fail_not_exist(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     value=123.5,
                                     unit="ms",
                                     dimensions=["model_name", "host"],
                                     metric_type=MetricTypes.gauge)

        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.get_metric(MetricTypes.gauge, "new_metric", "dim1:dim3")

        assert str(exc_info.value) == "'Metric key [gauge]-[new_metric]-[dim1:dim3] does not exist.'"

    def test_get_metric_fail_invalid_type(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.get_metric(["type-new_metric-dim1:dim3"], None, None)

        assert str(exc_info.value) == "metric_type must be MetricTypes enum, metric_name must be a str."

    def test_get_metric_fail_invalid_name(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.get_metric(MetricTypes.gauge, None, None)

        assert str(exc_info.value) == "metric_type must be MetricTypes enum, metric_name must be a str."

    def test_get_metric_fail_invalid_dims(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.get_metric(MetricTypes.gauge, "something", None)

        assert str(exc_info.value) == "'Metric key [gauge]-[something]-[None] does not exist.'"


class TestParseYaml:

    def test_parse_expected_yaml(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        expected_dict = {'dimensions': {'Level': 'Model',
                                        'ModelName': 'my_tc',
                                        'host': 'example_host_name',
                                        'model_name': 'example_model_name'},
                         'model_metrics': {'counter': [{'InferenceTimeInMS': {'dimensions': ['ModelName',
                                                                                             'Level'],
                                                                              'unit': 'ms'}},
                                                       {'NumberOfMetrics': {'dimensions': ['model_name', 'host'],
                                                                            'unit': 'count'}}],
                                           'gauge': [{'GaugeModelMetricNameExample': {'dimensions': ['ModelName',
                                                                                                     'Level'],
                                                                                      'unit': 'ms'}}],
                                           'histogram': [
                                               {'HistogramModelMetricNameExample': {'dimensions': ['ModelName',
                                                                                                   'Level'],
                                                                                    'unit': 'ms'}}]},
                         'ts_metrics': {'counter': [{'name': {'dimensions': ['model_name', 'host'],
                                                              'unit': 'ms'}}],
                                        'gauge': [{'name': {'dimensions': ['model_name', 'host'],
                                                            'unit': 'ms'}}],
                                        'histogram': [{'name': {'dimensions': ['model_name', 'host'],
                                                                'unit': 'ms'}}]}}
        assert expected_dict == metrics_cache_obj._parsed_file

    def test_yaml_file_passing(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")

        assert metrics_cache_obj.file == "metrics.yaml"

    def test_yaml_file_none_fail(self):
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            MetricsCacheYaml(None, None, None)
        assert str(exc_info.value) == "File None does not exist."

    def test_yaml_file_non_yaml_extension_fail(self):
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metric_cache_unit_test.py")
        assert str(exc_info.value) == "Inputted file metric_cache_unit_test.py " \
                                      "does not have a valid yaml file extension."

    def test_yaml_file_non_exist_fail(self):
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "doesnt_exist.yaml")
        assert str(exc_info.value) == "File doesnt_exist.yaml does not exist."

    def test_parse_yaml_errors(self):
        with pytest.raises(merrors.MetricsCachePyYamlError) as exc_info:
            MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metric_errors.yaml")
        assert str(exc_info.value) == 'Error parsing file: Error parsing file metric_errors.yaml: ' \
                                      'while parsing a block mapping\n ' \
                                      ' in "metric_errors.yaml", line 1, column 1\n' \
                                      "expected <block end>, but found '<block mapping start>'\n" \
                                      '  in "metric_errors.yaml", line 51, column 3'


class TestParseModelMetrics:
    def test_pass_parse_model_metrics(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        assert isinstance(metrics_cache_obj._parse_specific_metric(), dict)

    def test_pass_parse_ts_metrics(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        assert isinstance(metrics_cache_obj._parse_specific_metric("ts_metrics"), dict)

    def test_fail_parse_model_metrics(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_wo_model_metrics.yaml")
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj._parse_specific_metric()
        assert str(exc_info) == '<ExceptionInfo MetricsCacheKeyError("\'model_metrics\' key not found in yaml ' \
                                'file - \'model_metrics\'") tblen=2>'

    def test_fail_parse_none(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_wo_model_metrics.yaml")
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj._parse_specific_metric(None)
        assert str(exc_info) == '<ExceptionInfo MetricsCacheKeyError("\'None\' key not found in yaml file - ' \
                                'None") tblen=2>'

    def test_fail_parse_missing_fields(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_model_empty.yaml")
        assert metrics_cache_obj._parse_specific_metric() is None


class TestYamlCacheUtil:
    def test_yaml_to_cache_util_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        assert list(metrics_cache_obj.cache.keys()) == \
               ['[counter]-[InferenceTimeInMS]-[ModelName:my_tc,Level:Model]',
                '[counter]-[NumberOfMetrics]-[model_name:example_model_name,host:example_host_name'
                ',ModelName:ModelNameExample,Level:Model]',
                '[gauge]-[GaugeModelMetricNameExample]-[ModelName:my_tc,Level:Model]',
                '[histogram]-[HistogramModelMetricNameExample]-[ModelName:my_tc,Level:Model]']

    def test_yaml_to_cache_empty_dims(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_empty_dims.yml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        print(model_metrics_table)
        metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        assert list(metrics_cache_obj.cache.keys()) == \
               ['[counter]-[InferenceTimeInMS]-[ModelName:ModelNameExample,Level:Model]',
                '[counter]-[NumberOfMetrics]-[model_name:example_model_name,host:example_host_name'
                ',ModelName:ModelNameExample,Level:Model]',
                '[gauge]-[GaugeModelMetricNameExample]-[ModelName:my_tc,Level:Model]',
                '[histogram]-[HistogramModelMetricNameExample]-[ModelName:ModelNameExample,Level:Model]',
                '[histogram]-[AnotherHistogram]-[ModelName:ModelNameExample,Level:Model]']

    def test_yaml_to_cache_util_fail_missing_fields(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_missing_types.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        assert str(exc_info) == '<ExceptionInfo MetricsCacheKeyError("Key not found: \'unit\'") tblen=2>'

    def test_yaml_to_cache_util_fail_none(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_missing_types.yaml")
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj._yaml_to_cache_util(None)
        assert str(exc_info) == "<ExceptionInfo MetricsCacheTypeError('None section is None and does not " \
                                "exist') tblen=2>"

    def test_yaml_to_cache_util_fail_empty_section(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_model_empty.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        assert str(exc_info) == "<ExceptionInfo MetricsCacheTypeError('None section is None and does not " \
                                "exist') tblen=2>"



class TestYamlCache:
    def test_yaml_to_cache_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.parse_yaml_to_cache()
        assert list(
            metrics_cache_obj.cache.keys()) == ['[counter]-[InferenceTimeInMS]-['
                                                'ModelName:my_tc,Level:Model]',
                                                '[counter]-[NumberOfMetrics]-'
                                                '[model_name:example_model_name,host:example_host_name,'
                                                'ModelName:ModelNameExample,Level:Model]',
                                                '[gauge]-[GaugeModelMetricNameExample]-[ModelName:my_tc,Level:Model]',
                                                '[histogram]-[HistogramModelMetricNameExample]-'
                                                '[ModelName:my_tc,Level:Model]']


class TestManualAddMetricDimensions:
    def test_dimensions_metric_add_dimensions_custom_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric("TempName", 13.3, "count", dimensions=[Dimension("MyDim", "MyDimValue")],
                                     metric_type=MetricTypes.counter
                                     )
        assert list(metrics_cache_obj.cache.keys()) == ["[counter]-[TempName]-[MyDim:MyDimValue,"
                                                        "ModelName:ModelNameExample,Level:Model]"]

    def test_dimensions_metric_add_dimensions_yaml_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric("TempName", 13.3, "count", dimensions=["model_name"],
                                     metric_type=MetricTypes.counter)
        assert list(metrics_cache_obj.cache.keys()) == ["[counter]-[TempName]-[model_name:example_model_name,"
                                                        "ModelName:ModelNameExample,Level:Model]"]

    def test_dimensions_metric_add_dimensions_and_yaml_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_metric("TempName", 13.3, "count", dimensions=["model_name"],
                                     metric_type=MetricTypes.counter)
        metrics_cache_obj.parse_yaml_to_cache()
        assert list(metrics_cache_obj.cache.keys()) == ['[counter]-[TempName]-[model_name:example_model_name,'
                                                        'ModelName:ModelNameExample,Level:Model]',
                                                        '[counter]-[InferenceTimeInMS]-[ModelName:my_tc,Level:Model]',
                                                        '[counter]-[NumberOfMetrics]-'
                                                        '[model_name:example_model_name,host:example_host_name,'
                                                        'ModelName:ModelNameExample,Level:Model]',
                                                        '[gauge]-[GaugeModelMetricNameExample]-'
                                                        '[ModelName:my_tc,Level:Model]',
                                                        '[histogram]-[HistogramModelMetricNameExample]-'
                                                        '[ModelName:my_tc,Level:Model]']

    def test_dimensions_metric_add_dimensions_yaml_nonexistent(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        with pytest.raises(KeyError) as exc_info:
            metrics_cache_obj.add_metric("TempName", 13.3, "count", dimensions=["not_existing"], metric_type="counter")
        assert str(exc_info.value) == '"Dimension not found: \'not_existing\'"'

    def test_dimensions_metric_dimension_not_present(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_mismatching_dims.yaml")
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.parse_yaml_to_cache()
        assert str(exc_info.value) == '"Key not found: \'fake_model\'"'

    def test_dimensions_metric_dimension_warning_name_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("Temp-Name", 13.3, "count", dimensions=["model_name"],
                                     metric_type=MetricTypes.counter)

        assert "There is a '-' symbol found in Temp-Name argument. Please refrain from using the '-' " \
               "as it is used as the delimiter in the Metric object string." in caplog.text

    def test_dimensions_metric_dimension_warning_unit_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", 13.3, "count-unit", dimensions=["model_name"],
                                     metric_type=MetricTypes.counter)

        assert "There is a '-' symbol found in count-unit argument. Please refrain from using the '-' as it is used " \
               "as the delimiter in the Metric object string.\n" in caplog.text

    def test_dimensions_metric_dimension_warning_dim_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", 13.3, "count", dimensions=[Dimension("hello-world", "foo")],
                                     metric_type=MetricTypes.counter)

        assert "There is a '-' symbol found in hello-world:foo argument. Please refrain from using the '-' as it " \
               "is used as the delimiter in the Metric object string.\n" in caplog.text

    def test_dimensions_metric_dimension_warning_dim_two_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("TempName", 13.3, "count", dimensions=[Dimension("helloworld", "foo-bar")],
                                     metric_type=MetricTypes.counter)

        assert "There is a '-' symbol found in helloworld:foo-bar argument. Please refrain from using the '-' as " \
               "it is used as the delimiter in the Metric object string.\n" in caplog.text

    def test_dimensions_metric_bracket_naming_convention(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics_mismatching_dims.yaml")
        metrics_cache_obj.add_metric("[TempName]", 13.3, "count", dimensions=[Dimension("helloworld", "foobar")],
                                     metric_type=MetricTypes.counter)
        phrases = ["There is a '[' symbol found in [TempName] argument. Please refrain from using the '[' "
                   "as it is used as the delimiter in the Metric object string.",
                   "There is a ']' symbol found in [TempName] argument. Please refrain from using the "
                   "']' as it is used as the delimiter in the Metric object string."
                   ]
        for phrase in phrases:
            assert phrase in caplog.text


class TestAdditionalMetricMethods:
    def test_add_counter_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_counter("CounterName", 12)
        print(metrics_cache_obj.cache)
        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "CounterName",
                                              "ModelName:ModelNameExample,Level:Model")
        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:ModelNameExample"
            if i == 1:
                assert dimension.__str__() == "Level:Model"
        assert metric.value == 12
        assert metric.name == "CounterName"
        assert metric.metric_type == "counter"

        metrics_cache_obj.add_counter("CounterName", 25)
        print(metrics_cache_obj.cache)
        assert metric.value == 37

        metric.value = 13
        assert metric.value == 13

        metric.update(14)
        assert metric.value == 27

    def test_add_time_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_time("TimeName", 14)
        metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "TimeName", "ModelName:ModelNameExample,Level:Model")

        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:ModelNameExample"
            if i == 1:
                assert dimension.__str__() == "Level:Model"

        assert metric.value == 14
        assert metric.name == "TimeName"
        assert metric.metric_type == "gauge"

        metrics_cache_obj.add_time("TimeName", 25)
        assert metric.value == 25

        metric.update(17)
        assert metric.value == 17

    def test_add_time_diff_type_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_time("TimeName", 14, metric_type=MetricTypes.histogram)
        metric = metrics_cache_obj.get_metric(MetricTypes.histogram, "TimeName",
                                              "ModelName:ModelNameExample,Level:Model")
        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:ModelNameExample"
            if i == 1:
                assert dimension.__str__() == "Level:Model"

        assert metric.value == 14
        assert metric.name == "TimeName"
        assert metric.metric_type == "histogram"

        metrics_cache_obj.add_time("TimeName", 25, metric_type=MetricTypes.histogram)
        assert metric.value == 25

        metric.update(1)
        assert metric.value == 1

    def test_add_size_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_size("SizeName", 11)
        metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "SizeName", "ModelName:ModelNameExample,Level:Model")

        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:ModelNameExample"
            if i == 1:
                assert dimension.__str__() == "Level:Model"

        assert metric.value == 11
        assert metric.name == "SizeName"
        assert metric.metric_type == "gauge"

        metrics_cache_obj.add_size("SizeName", 25)
        assert metric.value == 25

        metric.update(1)
        assert metric.value == 1

    def test_add_size_diff_metric_type_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "ModelNameExample", "metrics.yaml")
        metrics_cache_obj.add_size("SizeName", 11, metric_type=MetricTypes.counter)
        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "SizeName", "ModelName:ModelNameExample,Level:Model")

        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:ModelNameExample"
            if i == 1:
                assert dimension.__str__() == "Level:Model"

        assert metric.value == 11
        assert metric.name == "SizeName"
        assert metric.metric_type == "counter"

        metrics_cache_obj.add_size("SizeName", 5, metric_type=MetricTypes.counter)
        assert metric.value == 16

        metric.update(1)
        assert metric.value == 17

    def test_add_percent_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        metrics_cache_obj.add_percent("PercentName", 12)
        metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "PercentName", "ModelName:Foo,Level:Model")

        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:Foo"
            if i == 1:
                assert dimension.__str__() == "Level:Model"

        assert metric.value == 12
        assert metric.name == "PercentName"
        assert metric.metric_type == "gauge"

        metrics_cache_obj.add_size("PercentName", 5)
        assert metric.value == 5

        metric.update(1)
        assert metric.value == 1

    def test_add_percent_diff_metric_type_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        metrics_cache_obj.add_percent("PercentName", 12, metric_type=MetricTypes.histogram)
        metric = metrics_cache_obj.get_metric(MetricTypes.histogram, "PercentName", "ModelName:Foo,Level:Model")

        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "ModelName:Foo"
            if i == 1:
                assert dimension.__str__() == "Level:Model"

        assert metric.value == 12
        assert metric.name == "PercentName"
        assert metric.metric_type == "histogram"

        metrics_cache_obj.add_size("PercentName", 5)
        metric_two = metrics_cache_obj.get_metric(MetricTypes.gauge, "PercentName", "ModelName:Foo,Level:Model")

        assert metric_two.value == 5

        metrics_cache_obj.add_percent("PercentName", 10, metric_type=MetricTypes.histogram)
        assert metric.value == 10

        metric.update(1)
        assert metric.value == 1

    def test_add_error_pass(self):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        metrics_cache_obj.add_error("ErrorName", 53)
        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "ErrorName", "Level:Error")

        for i, dimension in enumerate(metric.dimensions):
            if i == 0:
                assert dimension.__str__() == "Level:Error"

        assert metric.value == 53
        assert metric.name == "ErrorName"
        assert metric.metric_type == "counter"

        metrics_cache_obj.add_error("ErrorName", 10)
        assert metric.value == 63

        metric.update(-51)
        assert metric.value == 12


class TestEmitMetrics:
    def test_emit_metrics_list_none(self, caplog):
        emit_metrics(None)
        assert "Metrics 'None' are not valid." in caplog.text

    def test_emit_metrics_list_empty(self, caplog):
        metrics = []
        emit_metrics(metrics)
        assert "Metrics '[]' are not valid." in caplog.text

    def test_emit_metrics_list_single(self, caplog):
        caplog.set_level("INFO")
        metrics = ["some_string"]
        emit_metrics(metrics)
        assert "'some_string' is not a valid Metric object." in caplog.text

    def test_emit_metrics_list_multiple(self, caplog):
        caplog.set_level("INFO")
        metrics = ["some_string", "another_string"]
        emit_metrics(metrics)
        assert "'some_string' is not a valid Metric object." in caplog.text
        assert "'another_string' is not a valid Metric object." in caplog.text

    def test_emit_metrics_dict_empty(self, caplog):
        emit_metrics({})
        assert "Metrics '{}' are not valid." in caplog.text

    def test_emit_metrics_dict_single(self, caplog):
        caplog.set_level("INFO")
        metrics = {"Metric_name": "Metric_STRING"}
        emit_metrics(metrics)
        assert "'Metric_STRING' is not a valid Metric object." in caplog.text

    def test_emit_metrics_dict_multiple(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        metrics_cache_obj.parse_yaml_to_cache()

        for i, metric in enumerate(list(metrics_cache_obj.cache.values())):
            inc = i + 1
            metric.update(inc + (inc / 10))

        caplog.set_level("INFO")
        emit_metrics(metrics_cache_obj.cache)

        assert "[METRICS]InferenceTimeInMS.Milliseconds:1.1|#ModelName:my_tc,Level:Model" \
               "|#hostname:" in caplog.text
        assert "[METRICS]NumberOfMetrics.Count:2.2|#model_name:example_model_name,host:example_host_name," \
               "ModelName:Foo,Level:Model|" \
               "#hostname:" in caplog.text
        assert "[METRICS]GaugeModelMetricNameExample.Milliseconds:3.3|#ModelName:my_tc,Level:Model" \
               "|#hostname:" in caplog.text
        assert "[METRICS]HistogramModelMetricNameExample.Milliseconds:4.4|#ModelName:my_tc,Level:Model" \
               "|#hostname:" in caplog.text

    def test_emit_metrics_reset(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        metrics_cache_obj.parse_yaml_to_cache()

        for i, metric in enumerate(list(metrics_cache_obj.cache.values())):
            inc = i + 1
            metric.update(inc + (inc / 10))

        caplog.set_level("INFO")
        caplog.clear()
        emit_metrics(metrics_cache_obj.cache)
        assert "[METRICS]InferenceTimeInMS.Milliseconds:1.1|#ModelName:my_tc,Level:Model|" in caplog.text
        assert "[METRICS]NumberOfMetrics.Count:2.2|#model_name:example_model_name,host:example_host_name" in caplog.text
        assert "[METRICS]GaugeModelMetricNameExample.Milliseconds:3.3|#ModelName:my_tc,Level:Model" in caplog.text
        assert "[METRICS]HistogramModelMetricNameExample.Milliseconds:4.4|#ModelName:my_tc,Level:Model|" in caplog.text

        caplog.clear()
        emit_metrics(metrics_cache_obj.cache)  # ensure there are no metrics being empty since they should all be reset
        assert "" == caplog.text

    def test_emit_metrics_metric_single(self, caplog):
        metric = Metric(name="NewMetric", value=12, unit="ms", dimensions=[Dimension("hello", "world")],
                        metric_type="counter")
        metric.update(2.5)
        caplog.set_level("INFO")
        emit_metrics(metric)
        assert "[METRICS]NewMetric.Milliseconds:14.5|#hello:world|#hostname:" in caplog.text


class TestIncrementDecrementMetrics:
    def test_add_counter_dimensions_none(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        dimensions = None

        # Create a counter with name 'LoopCount' and dimensions, initial value
        metrics_cache_obj.add_counter('LoopCount', 1, None, dimensions)

        # Increment counter by 2
        metrics_cache_obj.add_counter('LoopCount', 2, None, dimensions)

        # Decrement counter by 1
        metrics_cache_obj.add_counter('LoopCount', -1, None, dimensions)

        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "LoopCount", "ModelName:Foo,Level:Model")

        assert metric.value == 2

    def test_add_counter_dimensions_empty(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        dimensions = []

        # Create a counter with name 'LoopCount' and dimensions, initial value
        metrics_cache_obj.add_counter('LoopCount', 1, None, dimensions)

        # Increment counter by 2
        metrics_cache_obj.add_counter('LoopCount', 2, None, dimensions)

        # Decrement counter by 1
        metrics_cache_obj.add_counter('LoopCount', -1, None, dimensions)

        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "LoopCount", "ModelName:Foo,Level:Model")
        assert metric.value == 2

    def test_add_counter_dimensions_filled(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        dimensions = [Dimension("foo", "bar")]

        # Create a counter with name 'LoopCount' and dimensions, initial value
        metrics_cache_obj.add_counter('LoopCount', 1, None, dimensions)

        # Increment counter by 2
        metrics_cache_obj.add_counter('LoopCount', 2, None, dimensions)

        # Decrement counter by 1
        metrics_cache_obj.add_counter('LoopCount', -1, None, dimensions)

        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "LoopCount", "foo:bar,ModelName:Foo,Level:Model")
        assert metric.value == 2

    def test_add_error_dimensions_filled(self, caplog):
        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics.yaml")
        dimensions = [Dimension("foo", "bar")]

        # Create a counter with name 'LoopCount' and dimensions, initial value
        metrics_cache_obj.add_error('LoopCountError', 5, dimensions)

        # Increment counter by 2
        metrics_cache_obj.add_error('LoopCountError', 1, dimensions)

        # Decrement counter by 1
        metrics_cache_obj.add_error('LoopCountError', -2, dimensions)

        metric = metrics_cache_obj.get_metric(MetricTypes.counter, "LoopCountError", "foo:bar,Level:Error")

        assert len(metrics_cache_obj.cache) == 1
        assert metric.value == 4

    def test_add_counter_existing_implementation(self, caplog):
        metrics_cache_obj = MetricsStore(uuid.uuid4(), "Foo")
        dimensions = [Dimension("foo", "bar")]

        # Create a counter with name 'LoopCount' and dimensions, initial value
        metrics_cache_obj.add_counter('LoopCountError', 1, None, dimensions)

        # Increment counter by 2
        metrics_cache_obj.add_counter('LoopCountError', 2, None, dimensions)

        # Decrement counter by 1
        metrics_cache_obj.add_counter('LoopCountError', -1, None, dimensions)

        assert len(metrics_cache_obj.cache) == 3


class TestAPIAndYamlParse:
    def test_yaml_then_api(self, caplog):
        dimensions = [Dimension("foo", "bar")]

        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics_api.yaml")
        metrics_cache_obj.parse_yaml_to_cache()

        metrics_cache_obj.add_counter('InferenceTimeInMS', 2.7)

        metrics_cache_obj.add_size('GaugeModelMetricNameExample', 25.12)
        # Create a counter with name 'LoopCount' and dimensions, initial value
        metrics_cache_obj.add_error('LoopCountError', 5, dimensions)
        metrics_cache_obj.add_size('GaugeModelMetricNameExample', 1.42)
        metrics_cache_obj.add_counter('InferenceTimeInMS', -2.7)

        assert len(metrics_cache_obj.cache) == 5

        counter_metric = metrics_cache_obj.get_metric(MetricTypes.counter, "InferenceTimeInMS",
                                                      "ModelName:Foo,Level:Model")

        assert counter_metric.value == 0
        assert counter_metric.is_updated is True

        gauge_metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "GaugeModelMetricNameExample",
                                                    "ModelName:Foo,Level:Model")

        assert gauge_metric.value == 1.42

        caplog.set_level("INFO")
        emit_metrics(metrics_cache_obj.cache)

        assert "InferenceTimeInMS.Milliseconds:0.0|#ModelName:Foo,Level:Model|" in caplog.text
        assert "GaugeModelMetricNameExample.Milliseconds:1.42|#ModelName:Foo,Level:Model|" in caplog.text
        assert "LoopCountError.:5|#foo:bar,Level:Error" in caplog.text

    def test_api_then_yaml(self, caplog):
        dimensions = [Dimension("foo", "bar")]

        metrics_cache_obj = MetricsCacheYaml(uuid.uuid4(), "Foo", "metrics_api.yaml")

        metrics_cache_obj.add_counter('InferenceTimeInMS', 5)
        metrics_cache_obj.add_counter('InferenceTimeInMS', -2)
        metrics_cache_obj.add_size('GaugeModelMetricNameExample', 0)

        # counter_metric = metrics_cache_obj.get_metric("[counter]-[InferenceTimeInMS]-[ModelName:Foo,Level:Model]")
        counter_metric = metrics_cache_obj.get_metric(MetricTypes.counter, "InferenceTimeInMS",
                                                      "ModelName:Foo,Level:Model")
        assert counter_metric.value == 3
        assert counter_metric.is_updated is True

        caplog.set_level("INFO")
        emit_metrics(metrics_cache_obj.cache)  # resets metrics
        caplog_list = caplog.text.split("\n")
        for entry in caplog_list:

            if "INFO" in entry:
                assert "InferenceTimeInMS.Count:3|#ModelName:Foo,Level:Model|" in entry

        assert counter_metric.value == 0
        assert counter_metric.is_updated is False
        assert "InferenceTimeInMS.Count:0|#ModelName:Foo,Level:Model|" in str(counter_metric)

        # now parsing the yaml file after adding some Metrics via API (this should never happen)

        metrics_cache_obj.parse_yaml_to_cache()
        assert len(metrics_cache_obj.cache) == 4

        counter_metric = metrics_cache_obj.get_metric(MetricTypes.counter, "InferenceTimeInMS",
                                                      "ModelName:Foo,Level:Model")
        assert counter_metric.value == 0

        assert counter_metric.is_updated is False

        gauge_metric = metrics_cache_obj.get_metric(MetricTypes.gauge, "GaugeModelMetricNameExample",
                                                    "ModelName:Foo,Level:Model")
        assert gauge_metric.value == 0

        assert "InferenceTimeInMS.Count:0|#ModelName:Foo,Level:Model|" in str(counter_metric)
        assert "InferenceTimeInMS.Milliseconds:0|#ModelName:Foo,Level:Model" not in str(counter_metric)


if __name__ == '__main__':
    pytest.main()
