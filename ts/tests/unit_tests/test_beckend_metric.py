import logging
import sys

import pytest
from ts.metrics.dimension import Dimension
from ts.metrics.metrics_store import MetricsStore
from ts.service import emit_metrics


def get_model_key(name, unit, req_id, model_name):
    dimensions = list()
    dimensions.append(Dimension("ModelName", model_name))
    dimensions.append(Dimension("Level", "Model"))
    dim_str = [name, unit, str(req_id)] + [str(d) for d in dimensions]
    return '-'.join(dim_str)


def get_error_key(name, unit):
    dimensions = list()
    dimensions.append(Dimension("Level", "Error"))
    dim_str = [name, unit, 'None'] + [str(d) for d in dimensions]
    return '-'.join(dim_str)


def test_metrics(caplog):
    """
    Test if metric classes methods behave as expected
    Also checks global metric service methods
    """
    # Create a batch of request ids
    caplog.set_level(logging.INFO)
    request_ids = {0: 'abcd', 1: "xyz", 2: "qwerty", 3: "hjshfj"}
    all_req_ids = ','.join(request_ids.values())
    model_name = "dummy model"

    # Create a metrics objects
    metrics = MetricsStore(request_ids, model_name)

    # Counter tests
    metrics.add_counter('CorrectCounter', 1, 1)
    test_metric = metrics.cache[get_model_key('CorrectCounter', 'count', 'xyz', model_name)]
    assert 'CorrectCounter' == test_metric.name
    metrics.add_counter('CorrectCounter', 1, 1)
    metrics.add_counter('CorrectCounter', 1, 3)
    metrics.add_counter('CorrectCounter', 1)
    test_metric = metrics.cache[get_model_key('CorrectCounter', 'count', all_req_ids, model_name)]
    assert 'CorrectCounter' == test_metric.name
    metrics.add_counter('CorrectCounter', 3)
    test_metric = metrics.cache[get_model_key('CorrectCounter', 'count', 'xyz', model_name)]
    assert test_metric.value == 2
    test_metric = metrics.cache[get_model_key('CorrectCounter', 'count', 'hjshfj', model_name)]
    assert test_metric.value == 1
    test_metric = metrics.cache[get_model_key('CorrectCounter', 'count', all_req_ids, model_name)]
    assert test_metric.value == 4
    # Check what is emitted is correct
    emit_metrics(metrics.store)

    assert "hjshfj" in caplog.text
    assert "ModelName:dummy model" in caplog.text

    # Adding other types of metrics
    # Check for time metric
    with pytest.raises(Exception) as e_info:
        metrics.add_time('WrongTime', 20, 1, 'ns')
    assert "the unit for a timed metric should be one of ['ms', 's']" == e_info.value.args[0]

    metrics.add_time('CorrectTime', 20, 2, 's')
    metrics.add_time('CorrectTime', 20, 0)
    test_metric = metrics.cache[get_model_key('CorrectTime', 'ms', 'abcd', model_name)]
    assert test_metric.value == 20
    assert test_metric.unit == 'Milliseconds'
    test_metric = metrics.cache[get_model_key('CorrectTime', 's', 'qwerty', model_name)]
    assert test_metric.value == 20
    assert test_metric.unit == 'Seconds'
    # Size based metrics
    with pytest.raises(Exception) as e_info:
        metrics.add_size('WrongSize', 20, 1, 'TB')
    assert "The unit for size based metric is one of ['MB','kB', 'GB', 'B']" == e_info.value.args[0]

    metrics.add_size('CorrectSize', 200, 0, 'GB')
    metrics.add_size('CorrectSize', 10, 2)
    test_metric = metrics.cache[get_model_key('CorrectSize', 'GB', 'abcd', model_name)]
    assert test_metric.value == 200
    assert test_metric.unit == 'Gigabytes'
    test_metric = metrics.cache[get_model_key('CorrectSize', 'MB', 'qwerty', model_name)]
    assert test_metric.value == 10
    assert test_metric.unit == 'Megabytes'

    # Check a percentage metric
    metrics.add_percent('CorrectPercent', 20.0, 3)
    test_metric = metrics.cache[get_model_key('CorrectPercent', 'percent', 'hjshfj', model_name)]
    assert test_metric.value == 20.0
    assert test_metric.unit == 'Percent'

    # Check a error metric
    metrics.add_error('CorrectError', 'Wrong values')
    test_metric = metrics.cache[get_error_key('CorrectError', '')]
    assert test_metric.value == 'Wrong values'
