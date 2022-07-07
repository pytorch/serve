"""
Unit testing for MetricsCache class and yaml parsing.
"""
import unittest
import sys
import yaml

sys.path.append("..")
from metric_cache import MetricsCaching


class TestAddMetrics(unittest.TestCase):

    def test_add_metric_passing(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["dim1", "dim2"],
                                     metric_type="type")

        self.assertTrue(True if "type-new_metric-dim1-dim2" in metrics_cache_obj.backend_cache else False)

    def test_add_metric_fail_metric_name(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")

        self.assertRaises(TypeError, metrics_cache_obj.add_metric(metric_name=42,
                                                                  unit="ms",
                                                                  dimensions=["dim1", "dim2"],
                                                                  metric_type="type"))

    def test_add_metric_fail_unit(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")

        self.assertRaises(TypeError, metrics_cache_obj.add_metric(metric_name="bar",
                                                                  unit=["foo"],
                                                                  dimensions=["dim1", "dim2"],
                                                                  metric_type="type"))

    def test_add_metric_fail_dimensions(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")

        self.assertRaises(TypeError, metrics_cache_obj.add_metric(metric_name="bar",
                                                                  unit="ms",
                                                                  dimensions="sink",
                                                                  metric_type="type"))

    def test_add_metric_fail_type(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")

        self.assertRaises(TypeError, metrics_cache_obj.add_metric(metric_name="bar",
                                                                  unit="ms",
                                                                  dimensions=["dim1", "dim2"],
                                                                  metric_type={"key": 42}))

    def test_add_metric_fail_type_unit(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")

        self.assertRaises(TypeError, metrics_cache_obj.add_metric(metric_name="bar",
                                                                  unit=["ms"],
                                                                  dimensions=["dim1", "dim2"],
                                                                  metric_type={"key": 42}))


class TestGetMetrics(unittest.TestCase):

    def test_get_metric_passing(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["dim1", "dim2"],
                                     metric_type="type")

        temp_metrics = metrics_cache_obj.get_metric("type-new_metric-dim1-dim2")
        self.assertEquals("new_metric", temp_metrics.name)

    def test_get_metric_fail_not_exist(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        metrics_cache_obj.add_metric(metric_name="new_metric",
                                     unit="ms",
                                     dimensions=["dim1", "dim2"],
                                     metric_type="type")

        with self.assertRaises(SystemExit):
            metrics_cache_obj.get_metric("type-new_metric-dim1-dim3")

    def test_get_metric_fail_invalid_type(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        with self.assertRaises(SystemExit):
            metrics_cache_obj.get_metric(["type-new_metric-dim1-dim3"])


class TestParseYaml(unittest.TestCase):

    def test_parse_expected_yaml(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        actual_dict = metrics_cache_obj._parse_yaml_file()
        expected_dict = {'dimensions': {'host': 'host', 'model_name': 'model_name'},
                         'model_metrics': {'counter': [{'dimensions': ['model_name', 'host'],
                                                        'name': None,
                                                        'unit': 'ms'}],
                                           'gauge': [{'dimensions': ['model_name', 'host'],
                                                      'name': None,
                                                      'unit': 'ms'}],
                                           'histogram': [{'dimensions': ['model_name', 'host'],
                                                          'name': None,
                                                          'unit': 'ms'}]},
                         'ts_metrics': {'counter': [{'dimensions': ['model_name', 'host'],
                                                     'name': None,
                                                     'unit': 'ms'}],
                                        'gauge': [{'dimensions': ['model_name', 'host'],
                                                   'name': None,
                                                   'unit': 'ms'}],
                                        'histogram': [{'dimensions': ['model_name', 'host'],
                                                       'name': None,
                                                       'unit': 'ms'}]}}
        self.assertEqual(actual_dict, expected_dict)

    def test_parse_medium_yaml(self):
        metrics_cache_obj = MetricsCaching("medium_10kb.yaml")
        self.assertIsNotNone(metrics_cache_obj._parse_yaml_file())

    def test_parse_yaml_errors(self):
        metrics_cache_obj = MetricsCaching("metric_errors.yaml")
        with self.assertRaises(SystemExit):
            metrics_cache_obj._parse_yaml_file()

    def test_parse_yaml_none(self):
        metrics_cache_obj = MetricsCaching(None)
        with self.assertRaises(SystemExit):
            metrics_cache_obj._parse_yaml_file()

    def test_parse_yaml_io_error(self):
        metrics_cache_obj = MetricsCaching("doesnt_exist.yaml")
        with self.assertRaises(SystemExit):
            metrics_cache_obj._parse_yaml_file()


class TestParseModelMetrics(unittest.TestCase):
    def test_pass_parse_model_metrics(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        self.assertTrue(isinstance(metrics_cache_obj._parse_specific_metric(), dict))

    def test_pass_parse_ts_metrics(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        self.assertTrue(isinstance(metrics_cache_obj._parse_specific_metric("ts_metrics"), dict))

    def test_fail_parse_model_metrics(self):
        metrics_cache_obj = MetricsCaching("metrics_wo_model_metrics.yaml")
        with self.assertRaises(SystemExit):
            metrics_cache_obj._parse_specific_metric()

    def test_fail_parse_none(self):
        metrics_cache_obj = MetricsCaching("metrics_wo_model_metrics.yaml")
        with self.assertRaises(SystemExit):
            metrics_cache_obj._parse_specific_metric(None)

    def test_fail_parse_missing_fields(self):
        metrics_cache_obj = MetricsCaching("metrics_model_empty.yaml")
        self.assertIsNone(metrics_cache_obj._parse_specific_metric())


class TestYamlCacheUtil(unittest.TestCase):
    def test_yaml_to_cache_util_pass(self):
        metrics_cache_obj = MetricsCaching("metrics.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        metrics_cache_obj._yaml_to_cache_util(model_metrics_table)
        self.assertEquals(['counter-None-model_name-host',
                           'gauge-None-model_name-host',
                           'histogram-None-model_name-host'], list(metrics_cache_obj.backend_cache.keys()))

    def test_yaml_to_cache_util_fail_missing_fields(self):
        metrics_cache_obj = MetricsCaching("metrics_missing_types.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with self.assertRaises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)

    def test_yaml_to_cache_util_fail_none(self):
        metrics_cache_obj = MetricsCaching("metrics_missing_types.yaml")
        with self.assertRaises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(None)

    def test_yaml_to_cache_util_fail_empty_section(self):
        metrics_cache_obj = MetricsCaching("metrics_model_empty.yaml")
        model_metrics_table = metrics_cache_obj._parse_specific_metric()
        with self.assertRaises(SystemExit):
            metrics_cache_obj._yaml_to_cache_util(model_metrics_table)


if __name__ == '__main__':
    unittest.main()
