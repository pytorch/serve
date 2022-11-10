"""
Unit testing for ts/metrics/metric_cache_abstract.py,
ts/metrics/metric_cache_yaml_impl.py, and emit_metrics() ts/service.py
"""
import os
import pytest
import uuid
import ts.metrics.metric_cache_errors as merrors

from ts.metrics.dimension import Dimension
from ts.metrics.caching_metric import CachingMetric
from ts.metrics.metric_cache_yaml_impl import MetricsCacheYamlImpl
from ts.metrics.metric_type_enum import MetricTypes

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestAddMetrics:
    def test_add_metric_passing(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.add_metric(
            metric_name="test_add_metric_passing",
            unit="ms",
            dimension_names=["ModelName", "Host"],
            metric_type=MetricTypes.GAUGE,
        )
        assert MetricTypes.GAUGE in metrics_cache_obj.cache.keys()
        assert "test_add_metric_passing" in metrics_cache_obj.cache[MetricTypes.GAUGE].keys()

    def test_add_metric_duplicate_passing(self, caplog):
        caplog.set_level("INFO")
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.add_metric(
            metric_name="test_add_metric_duplicate_passing",
            unit="ms",
            dimension_names=["ModelName"],
            metric_type=MetricTypes.GAUGE,
        )
        assert MetricTypes.GAUGE in metrics_cache_obj.cache.keys()
        assert "test_add_metric_duplicate_passing" in metrics_cache_obj.cache[MetricTypes.GAUGE].keys()
        metric = metrics_cache_obj.get_metric("test_add_metric_duplicate_passing", MetricTypes.GAUGE)
        metric.add_or_update(123.5, ["dummy_model"])
        assert "123.5" in caplog.text
        metrics_cache_obj.add_metric(
            metric_name="test_add_metric_duplicate_passing",
            unit="ms",
            dimension_names=["ModelName"],
            metric_type=MetricTypes.GAUGE,
        )
        metric = metrics_cache_obj.get_metric("test_add_metric_duplicate_passing", MetricTypes.GAUGE)
        metric.add_or_update(42.5, ["dummy_model"])
        assert "42.5" in caplog.text

    def test_add_metric_fail_metric_name(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(
                metric_name=42,
                unit="ms",
                dimension_names=["ModelName", "Host"],
                metric_type=MetricTypes.GAUGE,
            )
        assert str(exc_info.value) == "`metric_name` must be a str"

    def test_add_metric_fail_unit(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(
                metric_name="test_add_metric_fail_unit",
                unit=["foo"],
                dimension_names=["ModelName", "Host"],
                metric_type=MetricTypes.GAUGE,
            )
        assert str(exc_info.value) == "`unit` must be a str"

    def test_add_metric_fail_dimensions(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(
                metric_name="test_add_metric_fail_dimensions",
                unit="ms",
                dimension_names="ModelName",
                metric_type=MetricTypes.GAUGE,
            )
        assert str(exc_info.value) == "`dimension_names` should be a list of dimension name strings"

    def test_add_metric_fail_type(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(
                metric_name="test_add_metric_fail_type",
                unit="ms",
                dimension_names=["ModelName", "Host"],
                metric_type={"key": 42},
            )
        assert str(exc_info.value) == "`metric_type` must be a MetricTypes enum"

    def test_add_metric_fail_type_unit(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.add_metric(
                metric_name="bar",
                unit=["ms"],
                dimension_names=["model_name", "host"],
                metric_type={"key": 42},
            )
        assert str(exc_info.value) == "`unit` must be a str"


class TestGetMetrics:
    def test_get_metric_passing(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.add_metric(
            metric_name="test_get_metric_passing",
            unit="ms",
            dimension_names=["ModelName", "Host"],
            metric_type=MetricTypes.GAUGE,
        )
        temp_metrics = metrics_cache_obj.get_metric(
            "test_get_metric_passing",
            MetricTypes.GAUGE,
        )
        assert temp_metrics.metric_name == "test_get_metric_passing"
        assert isinstance(temp_metrics.dimension_names, list)
        assert temp_metrics.unit == "Milliseconds"

    def test_get_metric_invalid_metric_type(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.add_metric(
            metric_name="test_get_metric_invalid_metric_type",
            unit="ms",
            dimension_names=["ModelName", "Host"],
            metric_type=MetricTypes.GAUGE,
        )
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.get_metric("test_get_metric_invalid_metric_type", MetricTypes.COUNTER)
        assert (
            str(exc_info.value)
            == '"Metric of type \'MetricTypes.COUNTER\' and '
               'name \'test_get_metric_invalid_metric_type\' doesn\'t exist"'
        )

    def test_get_metric_fail_invalid_type(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.get_metric("type-new_metric-dim1:dim3", None)
        assert str(exc_info.value) == "`metric_type` must be a MetricTypes enum"

    def test_get_metric_fail_invalid_name(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            metrics_cache_obj.get_metric(None, MetricTypes.GAUGE)
        assert str(exc_info.value) == "`metric_name` must be a str"


class TestParseYaml:
    def test_parse_expected_yaml(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        expected_dict = {
            "mode": "prometheus",
            "dimensions":
            [
                "model_name",
                "host_name",
                "host"
            ],
            "ts_metrics":
            {
                "counter": [
                {
                    "name": "CounterTsMetricExample",
                    "unit": "ms",
                    "dimensions": ["model_name", "host_name"]
                }],
                "gauge": [
                {
                    "name": "GaugeTsMetricExample",
                    "unit": "ms",
                    "dimensions": ["model_name", "host_name"]
                }],
                "histogram": [
                {
                    "name": "HistogramTsMetricExample",
                    "unit": "ms",
                    "dimensions": ["model_name", "host_name"]
                }]
            },
            "model_metrics":
            {
                "counter": [
                {
                    "name": "InferenceTimeInMS",
                    "unit": "ms",
                    "dimensions": ["model_name", "host"]
                },
                {
                    "name": "NumberOfMetrics",
                    "unit": "count",
                    "dimensions": ["model_name", "host_name"]
                }],
                "gauge": [
                {
                    "name": "GaugeModelMetricNameExample",
                    "unit": "ms",
                    "dimensions": ["model_name", "host"]
                }],
                "histogram": [
                {
                    "name": "HistogramModelMetricNameExample",
                    "unit": "ms",
                    "dimensions": ["model_name", "host"]
                }]
            }
        }
        assert expected_dict == metrics_cache_obj._parsed_file

    def test_yaml_file_passing(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        assert metrics_cache_obj.config_file_path == os.path.join(dir_path, "metrics.yaml")

    def test_yaml_file_none_fail(self):
        with pytest.raises(merrors.MetricsCacheTypeError) as exc_info:
            MetricsCacheYamlImpl(None)
        assert "stat: path should be string, bytes, os.PathLike or integer, not NoneType" in str(exc_info.value)

    def test_yaml_file_non_yaml_extension_fail(self):
        with pytest.raises(merrors.MetricsCachePyYamlError) as exc_info:
            MetricsCacheYamlImpl(os.path.join(dir_path, "metric_cache_unit_test.py"))
        assert "Error parsing file" in str(exc_info.value)

    def test_yaml_file_non_exist_fail(self):
        with pytest.raises(merrors.MetricsCacheIOError) as exc_info:
            MetricsCacheYamlImpl(os.path.join(dir_path, "doesnt_exist.yaml"))
        assert "No such file or directory" in str(exc_info.value)


class TestParseModelMetrics:
    def test_pass_parse_model_metrics(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        assert isinstance(metrics_cache_obj._parse_metrics_section(), dict)

    def test_pass_parse_ts_metrics(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        assert isinstance(metrics_cache_obj._parse_metrics_section("ts_metrics"), dict)

    def test_fail_parse_model_metrics(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_wo_model_metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj._parse_metrics_section()
        assert (
            str(exc_info)
            == "<ExceptionInfo MetricsCacheKeyError(\"'model_metrics' key not found in yaml "
            "file: 'model_metrics'\") tblen=2>"
        )

    def test_fail_parse_none(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_wo_model_metrics.yaml"))
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj._parse_metrics_section(None)
        assert (
            str(exc_info)
            == "<ExceptionInfo MetricsCacheKeyError(\"'None' key not found in yaml file: "
            'None") tblen=2>'
        )

    def test_fail_parse_missing_fields(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_model_empty.yaml"))
        assert metrics_cache_obj._parse_metrics_section() is None


class TestYamlCacheInit:
    def test_yaml_to_cache_util_pass(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.initialize_cache()
        assert(len(metrics_cache_obj.cache.keys()) == 3)
        assert metrics_cache_obj.cache_keys() == [
            "counter:InferenceTimeInMS",
            "counter:NumberOfMetrics",
            "gauge:GaugeModelMetricNameExample",
            "histogram:HistogramModelMetricNameExample"
        ]

    def test_yaml_to_cache_empty_dims(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_empty_dims.yml"))
        metrics_cache_obj.initialize_cache()
        assert metrics_cache_obj.cache_keys() == [
            "counter:InferenceTimeInMS",
            "counter:NumberOfMetrics",
            "gauge:GaugeModelMetricNameExample",
            "histogram:HistogramModelMetricNameExample",
            "histogram:AnotherHistogram"
        ]
        for metric_type, metric in metrics_cache_obj.cache.items():
            for k, v in metric.items():
                if k in [
                    "HistogramModelMetricNameExample",
                    "AnotherHistogram",
                ]:
                    assert isinstance(v, CachingMetric)
                    assert v.dimension_names == []

    def test_yaml_to_cache_util_fail_missing_fields(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_missing_types.yaml"))
        with pytest.raises(merrors.MetricsCacheKeyError) as exc_info:
            metrics_cache_obj.initialize_cache()
        assert (
            str(exc_info)
            == "<ExceptionInfo MetricsCacheKeyError(\"Key not found in cache spec: 'unit'\") tblen=2>"
        )

    def test_yaml_to_cache_util_fail_empty_section(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_model_empty.yaml"))
        with pytest.raises(merrors.MetricsCacheValueError) as exc_info:
            metrics_cache_obj.initialize_cache()
        assert (
            str(exc_info)
            == "<ExceptionInfo MetricsCacheValueError('Missing `model_metrics` specification') tblen=2>"
        )


class TestManualAddMetricDimensions:
    def test_dimensions_metric_add_dimensions_custom_pass(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.add_metric(
            "TempName",
            "count",
            dimension_names=["MyDim", "MyDimValue"],
            metric_type=MetricTypes.COUNTER,
        )
        assert list(metrics_cache_obj.cache_keys()) == ["counter:TempName"]
        for metric_type, metric in metrics_cache_obj.cache.items():
            for k, v in metric.items():
                if k == "TempName":
                    assert isinstance(v, CachingMetric)
                    assert v.dimension_names == ["MyDim", "MyDimValue"]

    def test_dimensions_metric_add_dimensions_and_yaml_pass(self):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        metrics_cache_obj.add_metric(
            "TempName",
            "count",
            dimension_names=["ModelName"],
            metric_type=MetricTypes.COUNTER,
        )
        metrics_cache_obj.initialize_cache()
        assert metrics_cache_obj.cache_keys() == [
            "counter:TempName",
            "counter:InferenceTimeInMS",
            "counter:NumberOfMetrics",
            "gauge:GaugeModelMetricNameExample",
            "histogram:HistogramModelMetricNameExample"
        ]


class TestAdditionalMetricMethods:
    def test_add_counter_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        dimensions = [
            Dimension("ModelName", "test_add_counter_pass"),
            Dimension("Level", "Model")
        ]
        caplog.set_level("INFO")
        metrics_cache_obj.add_counter("CounterMetric", 14, dimensions=dimensions)
        metric = metrics_cache_obj.get_metric("CounterMetric", MetricTypes.COUNTER)
        assert metric.dimension_names == ["ModelName", "Level"]
        assert metric.metric_name == "CounterMetric"
        assert metric.metric_type == MetricTypes.COUNTER
        assert "[METRICS]CounterMetric.Count:14|#ModelName:test_add_counter_pass" in caplog.text

    def test_add_time_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        dimensions = [Dimension("ModelName", "test_add_time_pass")]
        caplog.set_level("INFO")
        metrics_cache_obj.add_time("TimeMetric", 17, unit="s", dimensions=dimensions)
        metric = metrics_cache_obj.get_metric("TimeMetric", MetricTypes.GAUGE)
        assert metric.metric_name == "TimeMetric"
        assert metric.metric_type == MetricTypes.GAUGE
        assert "[METRICS]TimeMetric.Seconds:17|#ModelName:test_add_time_pass" in caplog.text

    def test_add_time_diff_type_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        metrics_cache_obj.add_time("TimeMetric", 14, unit="ms",
                                   dimensions=[Dimension("Level", "Model")], metric_type=MetricTypes.HISTOGRAM)
        metric = metrics_cache_obj.get_metric("TimeMetric", MetricTypes.HISTOGRAM)
        assert metric.metric_name == "TimeMetric"
        assert metric.metric_type == MetricTypes.HISTOGRAM
        assert metric.dimension_names == ["Level"]
        assert "[METRICS]TimeMetric.Milliseconds:14|#Level:Model" in caplog.text
        metric.add_or_update(25, ["Model"])
        assert "[METRICS]TimeMetric.Milliseconds:25|#Level:Model" in caplog.text

    def test_add_size_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        dimensions = [Dimension("ModelName", "test_add_size_pass")]
        metrics_cache_obj.add_size("SizeMetric", 25, unit="GB", dimensions=dimensions)
        metric = metrics_cache_obj.get_metric("SizeMetric", MetricTypes.GAUGE)
        assert metric.dimension_names == ["ModelName", "Level"]
        assert "[METRICS]SizeMetric.Gigabytes:25|#ModelName:test_add_size_pass,Level:Error" in caplog.text

    def test_add_size_diff_metric_type_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        dimensions = [Dimension("Level", "test_add_size_diff_metric_type_pass")]
        metrics_cache_obj.add_size("SizeMetric", 5, unit="kB", dimensions=dimensions, metric_type=MetricTypes.COUNTER)
        metric = metrics_cache_obj.get_metric("SizeMetric", MetricTypes.COUNTER)
        assert "[METRICS]SizeMetric.Kilobytes:5|#Level:test_add_size_diff_metric_type_pass" in caplog.text

    def test_add_percent_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        dimensions = [Dimension("ModelName", "test_add_percent_pass")]
        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())
        request_id_map = {1: uid1, 2: uid2}
        metrics_cache_obj.set_request_ids(request_id_map)
        metrics_cache_obj.add_percent("PercentMetric", 11, uid1, dimensions)
        assert "[METRICS]PercentMetric.Percent:11|#ModelName:test_add_percent_pass,Level:Model|" in caplog.text
        assert str(uid1) in caplog.text
        metric = metrics_cache_obj.get_metric("PercentMetric", MetricTypes.GAUGE)
        metric.update(22, request_id=uid2, dimensions=dimensions)
        assert "[METRICS]PercentMetric.Percent:22|#ModelName:test_add_percent_pass,Level:Model|" in caplog.text
        assert str(uid2) in caplog.text

    def test_add_percent_diff_metric_type_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        metrics_cache_obj.add_percent("PercentMetric", 72,
                                      dimensions=[Dimension("Host", "hostname")], metric_type=MetricTypes.HISTOGRAM)
        assert "[METRICS]PercentMetric.Percent:72|#Host:hostname" in caplog.text

    def test_add_error_pass(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        metrics_cache_obj.add_error("ErrorName", 72, dimensions=[Dimension("ModelName", "hostname")])
        assert "[METRICS]ErrorName.unit:72|#ModelName:hostname" in caplog.text


class TestIncrementDecrementMetrics:
    def test_add_counter_dimensions_empty(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        metrics_cache_obj.add_counter("LoopCount", 7, uuid.uuid4(), [
            Dimension("ModelName", "test_add_counter_dimensions_empty")
        ])
        counter_metric = metrics_cache_obj.get_metric("LoopCount", MetricTypes.COUNTER)
        counter_metric.add_or_update(14, ["test_add_counter_dimensions_empty", "Host"])
        assert "LoopCount.Count:7|#ModelName:test_add_counter_dimensions_empty,Level:Error|" in caplog.text
        assert "LoopCount.Count:14|#ModelName:test_add_counter_dimensions_empty,Level:Host|" in caplog.text

    def test_add_counter_dimensions_filled(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        dimensions = [
            Dimension("ModelName", "model_name_a"),
            Dimension("Level", "level_a")
        ]
        metrics_cache_obj.add_counter("LoopCount", 71, dimensions=dimensions)
        metric = metrics_cache_obj.get_metric("LoopCount", MetricTypes.COUNTER)
        metric.add_or_update(19, ["model_name_b", "level_b"])
        assert "LoopCount.Count:71|#ModelName:model_name_a,Level:level_a|" in caplog.text
        assert "LoopCount.Count:19|#ModelName:model_name_b,Level:level_b|" in caplog.text

    def test_add_error_dimensions_filled(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics.yaml"))
        caplog.set_level("INFO")
        dimensions = [
            Dimension("ModelName", "model_name_a"),
            Dimension("Level", "level_a")
        ]
        metrics_cache_obj.add_error("LoopCountError", 2, dimensions=dimensions)
        metric = metrics_cache_obj.get_metric("LoopCountError", MetricTypes.COUNTER)
        metric.add_or_update(4, ["model_name_b", "level_b"])
        assert "LoopCountError.unit:2|#ModelName:model_name_a,Level:level_a|" in caplog.text
        assert "LoopCountError.unit:4|#ModelName:model_name_b,Level:level_b|" in caplog.text


class TestAPIAndYamlParse:
    def test_yaml_then_api(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_api.yaml"))
        metrics_cache_obj.initialize_cache()
        metrics_cache_obj.add_metric("InferenceTimeInMS", "ms", ["ModelName"])
        metrics_cache_obj.add_metric("GaugeModelMetricNameExample", "MB", ["Level"], MetricTypes.GAUGE)
        metrics_cache_obj.add_metric("LoopCountError", "")
        print(metrics_cache_obj.cache_keys())
        assert len(metrics_cache_obj.cache_keys()) == 5
        counter_metric = metrics_cache_obj.get_metric("InferenceTimeInMS", MetricTypes.COUNTER)
        gauge_metric = metrics_cache_obj.get_metric("GaugeModelMetricNameExample", MetricTypes.GAUGE)
        error_metric = metrics_cache_obj.get_metric("LoopCountError", MetricTypes.COUNTER)
        caplog.set_level("INFO")
        counter_metric.add_or_update(2.7, ["test_yaml_then_api"])
        gauge_metric.add_or_update(25.17, ["model"])
        error_metric.add_or_update(5)
        assert "InferenceTimeInMS.Milliseconds:2.7|#ModelName:test_yaml_then_api|" in caplog.text
        assert "GaugeModelMetricNameExample.Megabytes:25.17|#Level:model|" in caplog.text
        assert "LoopCountError.unit:5|#|" in caplog.text

    def test_api_then_yaml(self, caplog):
        metrics_cache_obj = MetricsCacheYamlImpl(os.path.join(dir_path, "metrics_api.yaml"))
        metrics_cache_obj.add_metric("InferenceTimeInMS", "count", ["ModelName"])
        metrics_cache_obj.add_metric("GaugeModelMetricNameExample", "MB", ["Level"], MetricTypes.GAUGE)
        counter_metric = metrics_cache_obj.get_metric("InferenceTimeInMS", MetricTypes.COUNTER)
        caplog.set_level("INFO")
        counter_metric.add_or_update(24.7, ["test_api_then_yaml"])
        assert "InferenceTimeInMS.Count:24.7|#ModelName:test_api_then_yaml|" in caplog.text
        counter_metric.add_or_update(42.5, ["updated"])
        assert "InferenceTimeInMS.Count:42.5|#ModelName:updated|" in caplog.text
        metrics_cache_obj.initialize_cache()
        assert len(metrics_cache_obj.cache_keys()) == 4
        counter_metric = metrics_cache_obj.get_metric("InferenceTimeInMS", MetricTypes.COUNTER)
        spec_metric = metrics_cache_obj.get_metric("NumberOfMetrics", MetricTypes.COUNTER)
        counter_metric.add_or_update(4.1, ["test_api_then_yaml", "model"])
        spec_metric.add_or_update(2, ["test_api_then_yaml"])
        assert "InferenceTimeInMS.Milliseconds:4.1|#model_name:test_api_then_yaml,level:model|" in caplog.text
        assert "NumberOfMetrics.Count:2|#model_name:test_api_then_yaml" in caplog.text


if __name__ == "__main__":
    pytest.main()
