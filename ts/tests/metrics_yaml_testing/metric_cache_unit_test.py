"""
Unit testing for MetricsCache class and yaml parsing.
"""
import pytest
from ts.metrics.dimension import Dimension
from ts.metrics.metric_cache import MetricsCache


class TestAddMetrics:

    def test_add_metric_passing(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["dim1", "dim2"],
                                     metric_type="type")

        assert "type-new_metric-dim1:dim2" in metrics_cache_obj.backend_cache

    def test_add_metric_dimensions_correct_obj_pass(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=[Dimension("hello", "world")],
                                     metric_type="type")

        for key, metric in metrics_cache_obj.backend_cache.items():

            for dimensions_list in metric.dimensions:
                assert isinstance(dimensions_list, Dimension)

    def test_add_metric_dimensions_correct_obj_fail(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")

        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name="new_metric",
                                         unit="ms",
                                         dimensions=Dimension("hello", "world"),
                                         metric_type="type")

    def test_add_metric_fail_metric_name(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name=42,
                                         unit="ms",
                                         dimensions=["dim1", "dim2"],
                                         metric_type="type")

    def test_add_metric_fail_unit(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit=["foo"],
                                         dimensions=["dim1", "dim2"],
                                         metric_type="type")

    def test_add_metric_fail_dimensions(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit="ms",
                                         dimensions="sink",
                                         metric_type="type")

    def test_add_metric_fail_dimensions_none(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit="ms",
                                         dimensions=None,
                                         metric_type="type")

    def test_add_metric_fail_type(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit="ms",
                                         dimensions=["dim1", "dim2"],
                                         metric_type={"key": 42})

    def test_add_metric_fail_type_unit(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(TypeError):
            metrics_cache_obj.add_metric(metric_name="bar",
                                         unit=["ms"],
                                         dimensions=["dim1", "dim2"],
                                         metric_type={"key": 42})


class TestGetMetrics:

    def test_get_metric_passing(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["dim1", "dim2"],
                                     metric_type="type")

        temp_metrics = metrics_cache_obj.get_metric("type-new_metric-dim1:dim2")
        assert "new_metric" == temp_metrics.name
        assert isinstance(temp_metrics.dimensions, list)
        assert "Milliseconds" == temp_metrics.unit
        assert 0 == temp_metrics.value

    def test_get_metric_fail_not_exist(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["dim1", "dim2"],
                                     metric_type="type")

        with pytest.raises(SystemExit):
            metrics_cache_obj.get_metric("type-new_metric-dim1:dim3")

    def test_get_metric_fail_invalid_type(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj.get_metric(["type-new_metric-dim1:dim3"])


class TestParseYaml:

    def test_parse_expected_yaml(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        actual_dict = metrics_cache_obj._parse_yaml_file()
        expected_dict = {'dimensions': {'host': 'host', 'model_name': 'model_name'},
                         'model_metrics': {'counter': [{'dimensions': ['model_name', 'host'],
                                                        'name': 'counter_name',
                                                        'unit': 'ms'}],
                                           'gauge': [{'dimensions': ['model_name', 'host'],
                                                      'name': 'gauge_name',
                                                      'unit': 'ms'}],
                                           'histogram': [{'dimensions': ['model_name', 'host'],
                                                          'name': 'histogram_name',
                                                          'unit': 'ms'}]},
                         'ts_metrics': {'counter': [{'dimensions': ['model_name', 'host'],
                                                     'name': 'counter_name',
                                                     'unit': 'ms'}],
                                        'gauge': [{'dimensions': ['model_name', 'host'],
                                                   'name': 'gauge_name',
                                                   'unit': 'ms'}],
                                        'histogram': [{'dimensions': ['model_name', 'host'],
                                                       'name': 'histogram_name',
                                                       'unit': 'ms'}]}}
        assert actual_dict == expected_dict

    def test_parse_yaml_errors(self):
        metrics_cache_obj = MetricsCache("metric_errors.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_yaml_file()

    def test_parse_yaml_none(self):
        metrics_cache_obj = MetricsCache(None)
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_yaml_file()

    def test_parse_yaml_io_error(self):
        metrics_cache_obj = MetricsCache("doesnt_exist.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_yaml_file()


class TestParseModelMetrics:
    def test_pass_parse_model_metrics(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        assert isinstance(metrics_cache_obj._parse_specific_metric(), dict)

    def test_pass_parse_ts_metrics(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        assert isinstance(metrics_cache_obj._parse_specific_metric("ts_metrics"), dict)

    def test_fail_parse_model_metrics(self):
        metrics_cache_obj = MetricsCache("metrics_wo_model_metrics.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_specific_metric()

    def test_fail_parse_none(self):
        metrics_cache_obj = MetricsCache("metrics_wo_model_metrics.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._parse_specific_metric(None)

    def test_fail_parse_missing_fields(self):
        metrics_cache_obj = MetricsCache("metrics_model_empty.yaml")
        assert metrics_cache_obj._parse_specific_metric() is None


class TestYamlCacheUtil:
    def test_yaml_to_cache_util_pass(self):
        metrics_cache_obj = MetricsCache("metrics.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        assert ['counter-counter_name-model_name:host',
                'gauge-gauge_name-model_name:host',
                'histogram-histogram_name-model_name:host'] == list(metrics_cache_obj.backend_cache.keys())

    def test_yaml_to_cache_util_fail_missing_fields(self):
        metrics_cache_obj = MetricsCache("metrics_missing_types.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)

    def test_yaml_to_cache_util_fail_none(self):
        metrics_cache_obj = MetricsCache("metrics_missing_types.yaml")
        with pytest.raises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(None)

    def test_yaml_to_cache_util_fail_empty_section(self):
        metrics_cache_obj = MetricsCache("metrics_model_empty.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with pytest.raises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)


if __name__ == '__main__':
    pytest.main()
