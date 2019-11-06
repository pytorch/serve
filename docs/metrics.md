# Metrics on Model Server

## Contents of this Document
* [Introduction](#introduction)
* [System metrics](#system-metrics)
* [Formatting](#formatting)
* [Custom Metrics API](#custom-metrics-api)

## Introduction
MMS collects system level metrics in regular intervals, and also provides an API for custom metrics to be collected. Metrics collected by metrics are logged and can be aggregated by metric agents.
The system level metrics are collected every minute. Metrics defined by the custom service code, can be collected per request or a batch of requests. MMS logs these two sets of metrics to different log files.
Metrics are collected by default at:
* System metrics - log_directory/mms_metrics.log
* Custom metrics - log directory/model_metrics.log

The location of log files and metric files can be configured at [log4j.properties](https://github.com/awslabs/mxnet-model-server/blob/master/frontend/server/src/main/resources/log4j.properties) file.


## System Metrics

|	Metric Name	|	Dimension	|	Unit	|	Semantics	|
|---|---|---|---|
|	CPUUtilization	|	host	|	percentage	|	cpu utillization on host	|
|	DiskAvailable	|	host	|	GB	|	disk available on host	|
|	DiskUsed	|	host	|	GB	|	disk used on host	|
|	DiskUtilization	|	host	|	percentage	|	disk used on host	|
|	MemoryAvailable	|	host	|	MB	|	memory available on host	|
|	MemoryUsed	|	host	|	MB	|	memory used on host	|
|	MemoryUtilization	|	host	|	percentage	|	memory used on host	|
|	Requests2XX	|	host	|	count	|	total number of requests that responded in 200-300 range	|
|	Requests4XX	|	host	|	count	|	total number of requests that responded in 400-500 range |
|	Requests5XX	|	host	|	count	|	total number of requests that responded above 500 |


## Formatting

The metrics emitted into log files by default, is in a [StatsD](https://github.com/etsy/statsd) like format.

```bash
CPUUtilization.Percent:0.0|#Level:Host|#hostname:my_machine_name
MemoryUsed.Megabytes:13840.328125|#Level:Host|#hostname:my_machine_name    
```

To enable metric logging in JSON format, we can modify the log formatter in [log4j.properties](https://github.com/awslabs/mxnet-model-server/blob/master/frontend/server/src/main/resources/log4j.properties), This is explained in the logging [document](https://github.com/awslabs/mxnet-model-server/blob/master/docs/logging.md).

to enable JSON formatting for metrics change it to 

```properties
log4j.appender.mms_metrics.layout = com.amazonaws.ml.mms.util.logging.JSONLayout
```

Once enabled the format emitted to logs, will look as follows

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

MMS enables the custom service code to emit metrics, that are then logged by the system

The custom service code is provided with a [context](https://github.com/awslabs/mxnet-model-server/blob/master/mms/context.py) of the current request.

Which has metrics object.

```python
# Access context metrics as follows
metrics = context.metrics
```
All metrics collected with in the context 

### Creating dimension object(s)

Dimensions for metrics can be defined as objects

```python
from mms.metrics import dimension

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
    def add_metric(name, value, idx=None, unit=None, dimensions=None):
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
metrics.add_metric('DistanceInKM', distance, 'km', dimensions)
```


### Add Time based metrics
Time based metrics can be added by invoking the following method

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

**Supported units**: ['ms', 's']

To add custom time based metrics

```python
# Add inference time
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size  is 1 for example
metrics.add_time('InferenceTime', end_time-start_time, None, 'ms', dimensions)
```

### Add Size based metrics
Size based metrics can be added by invoking the following method

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

Note that the default unit in this case is 'ms'

**Supported units**: ['MB', 'kB', 'GB']

To add custom size based metrics

```python
# Add Image size as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size is 1 for example
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

To add custom percentage based metrics

```python
# Add MemoryUtilization as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size  is 1 for example
metrics.add_percent('MemoryUtilization', utilization_percent, None, dimensions)
```

### Add Counter based metrics

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

To create , increment and decrement counter based metrics we can use the following calls
```python
# Add Loop Count as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size is 1 for example

# Create a counter with name 'LoopCount' and dimensions, initial value
metrics.add_counter('LoopCount', 1, None, dimensions)

# Increment counter by 2 
metrics.add_counter('LoopCount', 2 , None, dimensions)

# Decrement counter by 1
metrics.add_counter('LoopCount', -1, None, dimensions)

# Final counter value in this case is 2

```
