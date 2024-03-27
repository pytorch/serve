# [TorchServe Metrics](#torchserve-metrics)

## Contents of this document

* [Introduction](#introduction)
* [Getting Started](#getting-started-with-torchserve-metrics)
* [Metric Types](#metric-types)
* [Default Metrics](#default-metrics)
* [Custom Metrics API](#custom-metrics-api)

## Introduction

Torchserve metrics can be broadly classified into frontend and backend metrics.

#### Frontend metrics:
* API request status metrics
* Inference request metrics
* System utilization metrics

**Note:** System utilization metrics are collected periodically (default: once every minute)

#### Backend metrics:
* Default model metrics
* Custom model metrics

**Note:** Torchserve provides an [API](#custom-metrics-api) to collect custom model metrics.

Default frontend and backend metrics are shown in the [Default Metrics](#default-metrics) section.

Three metrics modes are supported, i.e `log`, `prometheus` and `legacy` with the default mode being `log`.
The metrics mode can be configured using the `metrics_mode` configuration option in `config.properties` or `TS_METRICS_MODE` environment variable.
For further details on `config.properties` and environment variable based configuration, refer [Torchserve Configuration](configuration.md) docs.

**Log Mode**

In `log` mode, metrics are logged and can be aggregated by metric agents.
Metrics are collected by default at the following locations in `log` mode:

* Frontend metrics - `log_directory/ts_metrics.log`
* Backend metrics - `log_directory/model_metrics.log`

The location of log files and metric files can be configured in the [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml) file

**Prometheus Mode**

In `prometheus` mode, metrics are made available in prometheus format via the [metrics API endpoint](metrics_api.md).

**Legacy Mode**

`legacy` mode enables backwards compatibility with Torchserve releases `<= 0.7.1`, where:
* `ts_inference_requests_total`, `ts_inference_latency_microseconds` and `ts_queue_latency_microseconds` are only available via the [metrics API endpoint](metrics_api.md) in prometheus format.
* Frontend metrics are logged to `log_directory/ts_metrics.log`
* Backend metrics are logged to `log_directory/model_metrics.log`

**Note:** To enable full backwards compatibility with releases `<= 0.7.1`, use legacy metrics mode with [Model Metrics Auto-Detection](#getting-started-with-torchserve-metrics) enabled.

## Getting Started with TorchServe Metrics

TorchServe defines metrics configuration in a [yaml](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml) file, including both frontend metrics (i.e. `ts_metrics`) and backend metrics (i.e. `model_metrics`).
When TorchServe is started, the metrics definition is loaded in the frontend and backend cache separately.
The backend emits metrics logs as they are updated. The frontend parses these logs and makes the corresponding metrics available either as logs or via the [metrics API endpoint](metrics_api.md) based on the `metrics_mode` configuration.

Dynamic updates to the metrics configuration file is not supported. In order to account for updates made to the metrics configuration file, Torchserve will need to be restarted.

By default, metrics that are not defined in the metrics configuration file will not be logged in the metrics log files or made available via the prometheus metrics API endpoint.
Backend model metrics can be `auto-detected` by setting `model_metrics_auto_detect` to `true` in `config.properties`
or using the `TS_MODEL_METRICS_AUTO_DETECT` environment variable. By default, model metrics auto-detection is disabled.

`Warning: Using auto-detection of backend metrics will have performance impact in the form of latency overhead, typically at model load and first inference for a given model.
This cold start behavior is because, it is during model load and first inference that new metrics are typically emitted by the backend and is detected and registered by the frontend.
Subsequent inferences could also see performance impact if new metrics are updated for the first time.
For use cases where multiple models are loaded/unloaded often, the latency overhead can be mitigated by specifying known metrics in the metrics configuration file, ahead of time.`

The `metrics.yaml` is formatted with Prometheus metric type terminology:

```yaml
dimensions: # dimension aliases
  - &model_name "ModelName"
  - &level "Level"

ts_metrics:  # frontend metrics
  counter:  # metric type
    - name: NameOfCounterMetric  # name of metric
      unit: ms  # unit of metric
      dimensions: [*model_name, *level]  # dimension names of metric (referenced from the above dimensions dict)
  gauge:
    - name: NameOfGaugeMetric
      unit: ms
      dimensions: [*model_name, *level]
  histogram:
    - name: NameOfHistogramMetric
      unit: ms
      dimensions: [*model_name, *level]

model_metrics:  # backend metrics
  counter:  # metric type
    - name: InferenceTimeInMS  # name of metric
      unit: ms  # unit of metric
      dimensions: [*model_name, *level]  # dimension names of metric (referenced from the above dimensions dict)
    - name: NumberOfMetrics
      unit: count
      dimensions: [*model_name]
  gauge:
    - name: GaugeModelMetricNameExample
      unit: ms
      dimensions: [*model_name, *level]
  histogram:
    - name: HistogramModelMetricNameExample
      unit: ms
      dimensions: [*model_name, *level]
```

Default metrics are provided in the [metrics.yaml](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml) file, but the user can either delete them to their liking / ignore them altogether, because these metrics will not be emitted unless they are updated.\
When adding custom `model_metrics` in the metrics configuration file, ensure to include `ModelName` and `Level` dimension names towards the end of the list of dimensions since they are included by default by the following custom metrics APIs:
[add_metric](#function-api-to-add-generic-metrics-with-default-dimensions), [add_counter](#add-counter-based-metrics),
[add_time](#add-time-based-metrics), [add_size](#add-size-based-metrics) or [add_percent](#add-percentage-based-metrics).

### Starting TorchServe Metrics

Whenever torchserve starts, the [backend worker](https://github.com/pytorch/serve/blob/master/ts/model_service_worker.py) initializes `service.context.metrics` with the [MetricsCache](https://github.com/pytorch/serve/blob/master/ts/metrics/metric_cache_yaml_impl.py) object. The `model_metrics` (backend metrics) section within the specified yaml file will be parsed, and Metric objects will be created based on the parsed section and added to the cache.

This is all done internally, so the user does not have to do anything other than specifying the desired yaml file.

*Users have the ability to parse other sections of the yaml file manually, but the primary purpose of this functionality is to
parse the backend metrics from the yaml file.*

***How It Works***

1. Create a `metrics.yaml` file to parse metrics from ***OR*** utilize the default [metrics.yaml](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml)


2. Set `metrics_config` argument equal to the yaml file path in the `config.properties` being used:
    ```properties
    ...
    ...
    workflow_store=../archive/src/test/resources/workflows
    metrics_config=/<path>/<to>/<metrics>/<file>/metrics.yaml
    ...
    ...
    ```

   If a `metrics_config` argument is not specified, the default yaml file will be used.


3. Set the metrics mode you would like to use using the `metrics_mode` configuration option in `config.properties` or `TS_METRICS_MODE` environment variable. If not set, `log` mode will be used by default.


4. Run torchserve and specify the path of the `config.properties` after the `ts-config` flag: (example using [Huggingface_Transformers](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers))

   ```torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --ncs --ts-config /<path>/<to>/<config>/<file>/config.properties```


5. Collect metrics depending on mode chosen.

    If `log` mode check :
    * Frontend metrics - `log_directory/ts_metrics.log`
    * Backend metrics - `log_directory/model_metrics.log`

    Else, if using `prometheus` mode, use the [Metrics API](metrics_api.md).

## Metric Types

TorchServe Metrics use [Metric Types](https://github.com/pytorch/serve/blob/master/ts/metrics/metric_type_enum.py)
that are in line with the [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types).

Metric types are an attribute of Metric objects.
Users will be restricted to the existing metric types when adding custom metrics.

```python
class MetricTypes(enum.Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
```

## Default Metrics

### Frontend Metrics

| Metric Name                       | Type    | Unit         | Dimensions                          | Semantics                                                                   |
|-----------------------------------|---------|--------------|-------------------------------------|-----------------------------------------------------------------------------|
| Requests2XX                       | counter | Count        | Level, Hostname                     | Total number of requests with response in 200-300 status code range         |
| Requests4XX                       | counter | Count        | Level, Hostname                     | Total number of requests with response in 400-500 status code range         |
| Requests5XX                       | counter | Count        | Level, Hostname                     | Total number of requests with response status code above 500                |
| ts_inference_requests_total       | counter | Count        | model_name, model_version, hostname | Total number of inference requests received                                 |
| ts_inference_latency_microseconds | counter | Microseconds | model_name, model_version, hostname | Total inference latency in Microseconds                                     |
| ts_queue_latency_microseconds     | counter | Microseconds | model_name, model_version, hostname | Total queue latency in Microseconds                                         |
| QueueTime                         | gauge   | Milliseconds | Level, Hostname                     | Time spent by a job in request queue in Milliseconds                        |
| WorkerThreadTime                  | gauge   | Milliseconds | Level, Hostname                     | Time spent in worker thread excluding backend response time in Milliseconds |
| WorkerLoadTime                    | gauge   | Milliseconds | WorkerName, Level, Hostname         | Time taken by worker to load model in Milliseconds                          |
| CPUUtilization                    | gauge   | Percent      | Level, Hostname                     | CPU utilization on host                                                     |
| MemoryUsed                        | gauge   | Megabytes    | Level, Hostname                     | Memory used on host                                                         |
| MemoryAvailable                   | gauge   | Megabytes    | Level, Hostname                     | Memory available on host                                                    |
| MemoryUtilization                 | gauge   | Percent      | Level, Hostname                     | Memory utilization on host                                                  |
| DiskUsage                         | gauge   | Gigabytes    | Level, Hostname                     | Disk used on host                                                           |
| DiskUtilization                   | gauge   | Percent      | Level, Hostname                     | Disk used on host                                                           |
| DiskAvailable                     | gauge   | Gigabytes    | Level, Hostname                     | Disk available on host                                                      |
| GPUMemoryUtilization              | gauge   | Percent      | Level, DeviceId, Hostname           | GPU memory utilization on host, DeviceId                                    |
| GPUMemoryUsed                     | gauge   | Megabytes    | Level, DeviceId, Hostname           | GPU memory used on host, DeviceId                                           |
| GPUUtilization                    | gauge   | Percent      | Level, DeviceId, Hostname           | GPU utilization on host, DeviceId                                           |

### Backend Metrics:

| Metric Name                       | Type  | Unit | Dimensions                 | Semantics                     |
|-----------------------------------|-------|------|----------------------------|-------------------------------|
| HandlerTime                       | gauge | ms   | ModelName, Level, Hostname | Time spent in backend handler |
| PredictionTime                    | gauge | ms   | ModelName, Level, Hostname | Backend prediction time       |

## Custom Metrics API

This is the API used in the backend handler to emit metrics. TorchServe enables the custom handler code to emit metrics that are then made available based on the configured `metrics_mode`.

**Example with custom handler showing [usage of custom metrics APIs](https://github.com/pytorch/serve/blob/master/examples/custom_metrics)**.

The [custom handler](../docs/custom_service.md) code is provided with a [context](https://github.com/pytorch/serve/blob/master/ts/context.py) of the current request consisting of a `metrics` object:
```python
# Access metrics object in context as follows
def initialize(self, context):
    metrics = context.metrics
```

**Note:** The custom metrics API is not to be confused with the [metrics API endpoint](metrics_api.md) which is a HTTP API that is used to fetch metrics in the prometheus format.

### Default dimensions

Metrics will have a couple of default dimensions if not already specified:
  * `ModelName: {name_of_model}`
  * `Level: Model`


### Create dimension object(s)

[Dimensions](https://github.com/pytorch/serve/blob/master/ts/metrics/dimension.py) for metrics can be defined as objects

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

### Add generic metrics

Generic metrics default to `COUNTER` metric type

#### Function API to add generic metrics without default dimensions

```python
    def add_metric_to_cache(
        self,
        metric_name: str,
        unit: str,
        dimension_names: list = [],
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ) -> CachingMetric:
        """
        Create a new metric and add into cache. Override existing metric if already present.

        Parameters
        ----------
        metric_name str
            Name of metric
        unit str
            unit can be one of ms, percent, count, MB, GB or a generic string
        dimension_names list
            list of dimension name strings for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        Returns
        -------
        newly created Metrics object
        """
```

[CachingMetric](https://github.com/pytorch/serve/blob/master/ts/metrics/caching_metric.py) APIs to update a metric

```python
    def add_or_update(
        self,
        value: int or float,
        dimension_values: list = [],
        request_id: str = "",
):
    """
    Update metric value, request id and dimensions

    Parameters
    ----------
    value : int, float
        metric to be updated
    dimension_values : list
        list of dimension value strings
    request_id : str
        request id to be associated with the metric
    """
```
```python
    def update(
        self,
        value: int or float,
        request_id: str = "",
        dimensions: list = [],
):
    """
    BACKWARDS COMPATIBILITY: Update metric value

    Parameters
    ----------
    value : int, float
        metric to be updated
    request_id : str
        request id to be associated with the metric
    dimensions : list
        list of Dimension objects
    """
```

```python
# Example usage
metrics = context.metrics
# Add metric
distance_metric = metrics.add_metric_to_cache(name='DistanceInKM', unit='km', dimension_names=[...])
# Update metric
distance_metric.add_or_update(value=distance, dimension_values=[...], request_id=context.get_request_id())
# OR
distance_metric.update(value=distance, request_id=context.get_request_id(), dimensions=[...])
```

**Note:** Calling `add_metric_to_cache` will not emit the metric, `add_or_update` will need to be called on the metric object as shown above.

#### Function API to add generic metrics with default dimensions

```python
    def add_metric(
        self,
        name: str,
        value: int or float,
        unit: str,
        idx: str = None,
        dimensions: list = [],
        metric_type: MetricTypes = MetricTypes.COUNTER,
    ):
        """
        Add a generic metric
            Default metric type is counter

        Parameters
        ----------
        name : str
            metric name
        value: int or float
            value of the metric
        unit: str
            unit of metric
        idx: str
            request id to be associated with the metric
        dimensions: list
            list of Dimension objects for the metric
        metric_type MetricTypes
            Type of metric Counter, Gauge, Histogram
        """
```

```python
# Example usage
metrics = context.metrics
metric = metrics.add_metric(name='DistanceInKM', value=10, unit='km', dimensions=[...])
```

### Add time-based metrics

Time-based metrics default to `GAUGE` metric type

```python
    def add_time(self, name: str, value: int or float, idx=None, unit: str = 'ms', dimensions: list = None,
                 metric_type: MetricTypes = MetricTypes.GAUGE):
        """
        Add a time based metric like latency, default unit is 'ms'
            Default metric type is gauge

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
            list of Dimension objects for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Time metrics
        """
```

**Note:** Default unit is `ms`

**Supported units**: `['ms', 's']`

```python
# Example usage
metrics = context.metrics
metrics.add_time(name='InferenceTime', value=end_time-start_time, idx=None, unit='ms', dimensions=[...])
```

### Add size-based metrics

Size-based metrics default to `GAUGE` metric type

```python
    def add_size(self, name: str, value: int or float, idx=None, unit: str = 'MB', dimensions: list = None,
                 metric_type: MetricTypes = MetricTypes.GAUGE):
        """
        Add a size based metric
            Default metric type is gauge

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
            list of Dimension objects for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Size metrics
        """
```

**Note:** Default unit is `MB`.

**Supported units**: `['MB', 'kB', 'GB', 'B']`

```python
# Example usage
metrics = context.metrics
metrics.add_size(name='SizeOfImage', value=img_size, idx=None, unit='MB', dimensions=[...])
```

### Add Percentage based metrics

Percentage-based metrics default to a `GAUGE` metric type

```python
    def add_percent(self, name: str, value: int or float, idx=None, dimensions: list = None,
                    metric_type: MetricTypes = MetricTypes.GAUGE):
        """
        Add a percentage based metric
            Default metric type is gauge

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of Dimension objects for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Percent metrics
        """
```

**Inferred unit**: `percent`

```python
# Example usage
metrics = context.metrics
metrics.add_percent(name='MemoryUtilization', value=utilization_percent, idx=None, dimensions=[...])
```

### Add counter-based metrics

Counter-based metrics default to `COUNTER` metric type

```python
    def add_counter(self, name: str, value: int or float, idx=None, dimensions: list = None):
        """
        Add a counter metric or increment an existing counter metric
            Default metric type is counter
        Parameters
        ----------
        name : str
            metric name
        value: int or float
            value of metric
        idx: int
            request_id index in batch
        dimensions: list
            list of Dimension objects for the metric
        """
```

```python
# Example usage
metrics = context.metrics
metrics.add_counter(name='CallCount', value=call_count, idx=None, dimensions=[...])
```

**Inferred unit**: `count`

### Getting a metric

Users can get a metric from the cache. The [CachingMetric](https://github.com/pytorch/serve/blob/master/ts/metrics/caching_metric.py) object is returned,
so the user can access the methods of CachingMetric to update the metric: (i.e. `CachingMetric.add_or_update(value, dimension_values)`, `CachingMetric.update(value, dimensions)`)

```python
    def get_metric(
        self,
        metric_name: str,
        metric_type: MetricTypes = MetricTypes.COUNTER,
) -> CachingMetric:
    """
    Create a new metric and add into cache

    Parameters
    ----------
    metric_name str
        Name of metric

    metric_type MetricTypes
        Type of metric Counter, Gauge, Histogram

    Returns
    -------
    Metrics object or MetricsCacheKeyError if not found
    """
```

```python
# Example usage
metrics = context.metrics
# Get metric
gauge_metric = metrics.get_metric(metric_name = "GaugeMetricName", metric_type = MetricTypes.GAUGE)
# Update metric
gauge_metric.add_or_update(value=gauge_metric_value, dimension_values=[...], request_id=context.get_request_id())
# OR
gauge_metric.update(value=gauge_metric_value, request_id=context.get_request_id(), dimensions=[...])
```
