# TorchServe Metrics

## Contents of this document

* [Introduction](#introduction)
* [System metrics](#system-metrics)
* [Formatting](#formatting)
* [Custom Metrics API](#custom-metrics-api)
* [Logging the custom metrics](#logging-the-custom-metrics)

## Introduction

TorchServe collects system level metrics in regular intervals, and also provides an API to collect custom metrics.
Metrics collected by metrics are logged and can be aggregated by metric agents.
The system level metrics are collected every minute. Metrics defined by the custom service code can be collected per request or per a batch of requests.
TorchServe logs these two sets of metrics to different log files.
Metrics are collected by default at:

* System metrics - `log_directory/ts_metrics.log`
* Custom metrics - `log directory/model_metrics.log`

The location of log files and metric files can be configured i in the [log4j.properties](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j.properties) file

## System Metrics

| Metric Name | Dimension | Unit | Semantics |
|---|---|---|---|
| CPUUtilization | host | percentage | CPU utilization on host |
| DiskAvailable | host | GB | disk available on host |
| DiskUsed | host | GB | disk used on host |
| DiskUtilization | host | percentage | disk used on host |
| MemoryAvailable | host | MB | memory available on host |
| MemoryUsed | host | MB | memory used on host |
| MemoryUtilization | host | percentage | memory utilization on host |
| Requests2XX | host | count | logged for every request responded in 200-300 status code range |
| Requests4XX | host |count | logged for every request responded in 400-500 status code range |
| Requests5XX | host | count | logged for every request responded with status code above 500 |

## Formatting

TorchServe emits metrics to log files by default. The metrics are formatted in a [StatsD](https://github.com/etsy/statsd) like format.

```bash
CPUUtilization.Percent:0.0|#Level:Host|#hostname:my_machine_name
MemoryUsed.Megabytes:13840.328125|#Level:Host|#hostname:my_machine_name   
```

To enable metric logging in JSON format, modify the log formatter in [log4j.properties](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j.properties). For information, see [Logging in Torchserve](https://github.com/pytorch/serve/blob/master/docs/logging.md).

To enable JSON formatting for metrics, change the following line in `log4j.properties`: 

```properties
log4j.appender.ts_metrics.layout = org.pytorch.serve.util.logging.JSONLayout
```

After you enable JSON log formatting, logs will look as follows:

```json
{
  "MetricName": "DiskAvailable",
  "Value": "108.15547180175781",
  "Unit": "Gigabytes",
  "Dimensions": [
    {
      "Name": "Level",
      "Value": "Host"
    }
  ],
  "HostName": "my_machine_name"
}
```

```json
{
  "MetricName": "DiskUsage",
  "Value": "124.13163757324219",
  "Unit": "Gigabytes",
  "Dimensions": [
    {
      "Name": "Level",
      "Value": "Host"
    }
  ],
  "HostName": "my_machine_name"
}

```

## Custom Metrics API

TorchServe enables the custom service code to emit metrics that are then logged by the system.

The custom service code is provided with a [context](https://github.com/pytorch/serve/blob/master/ts/context.py) of the current request with a metrics object:


```python
# Access context metrics as follows
metrics = context.metrics
```

All metrics are collected within the context.

### Create dimension object(s)

Dimensions for metrics can be defined as objects

```python
from ts.metrics.dimension import Dimension

# Dimensions are name value pairs
dim1 = Dimension(name, value)
dim2 = Dimension(some_name, some_value)
.
.
.
dimN= Dimension(name_n, value_n)

```

**NOTE:** Metric functions below accept a list of dimensions

### Add generic metrics

One can add metrics with generic units using the following function.

Function API

```python
    def add_metric(name, value, unit, idx=None, dimensions=None):
        """
        Add a metric which is generic with custom metrics

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric
        dimensions: list
            list of dimensions for the metric
        """
```

```python
# Add Distance as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size is 1 for example
metrics.add_metric('DistanceInKM', distance, 'km', dimensions=dimensions)
```

### Add time-based metrics

Add time-based by invoking the following method:

Function API

```python
    def add_time(name, value, idx=None, unit='ms', dimensions=None):
        """
        Add a time based metric like latency, default unit is 'ms'

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric,  default here is ms, s is also accepted
        dimensions: list
            list of dimensions for the metric
        """
```

Note that the default unit in this case is 'ms'

**Supported units**: `['ms', 's']`

To add custom time-based metrics:

```python
# Add inference time
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size  is 1 for example
metrics.add_time('InferenceTime', end_time-start_time, None, 'ms', dimensions)
```

### Add size-based metrics

Add size-based metrics by invoking the following method:

Function API

```python
    def add_size(name, value, idx=None, unit='MB', dimensions=None):
        """
        Add a size based metric

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric, default here is 'MB', 'kB', 'GB' also supported
        dimensions: list
            list of dimensions for the metric
        """
```

Note that the default unit in this case is milliseconds (ms).

**Supported units**: `['MB', 'kB', 'GB']`

To add custom size based metrics

```python
# Add Image size as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size 1
metrics.add_size('SizeOfImage', img_size, None, 'MB', dimensions)
```

### Add Percentage based metrics

Percentage based metrics can be added by invoking the following method

Function API

```python
    def add_percent(name, value, idx=None, dimensions=None):
        """
        Add a percentage based metric

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        """
```

To add custom percentage-based metrics:

```python
# Add MemoryUtilization as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size 1
metrics.add_percent('MemoryUtilization', utilization_percent, None, dimensions)
```

### Add counter-based metrics

Percentage based metrics can be added by invoking the following method

Function API

```python
    def add_counter(name, value, idx=None, dimensions=None):
        """
        Add a counter metric or increment an existing counter metric

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of dimensions for the metric
        """
```

To create, increment and decrement counter-based metrics we can use the following calls:

```python
# Add Loop Count as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size 1

# Create a counter with name 'LoopCount' and dimensions, initial value
metrics.add_counter('LoopCount', 1, None, dimensions)

# Increment counter by 2 
metrics.add_counter('LoopCount', 2 , None, dimensions)

# Decrement counter by 1
metrics.add_counter('LoopCount', -1, None, dimensions)

# Final counter value in this case is 2

```

### Log custom metrics

Following sample code can be used to log the custom metrics created in the model's custom handler:

```python
for metric in metrics.store:
    logger.info("[METRICS]%s", str(metric))
```

This custom metrics information is logged in the `model_metrics.log` file configured through [log4j.properties](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j.properties) file.
