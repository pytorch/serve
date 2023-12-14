# [TorchServe Metrics](#torchserve-metrics)

## Contents of this document

* [Introduction](#introduction)
* [Getting Started](#getting-started-with-torchserve-metrics)
* [Metric Types](#metric-types)
* [Metrics Formatting](#metrics-formatting)
* [Custom Metrics API](#custom-metrics-api)
* [Emitting custom metrics](#emitting-custom-metrics)
* [Metrics YAML Parsing and Metrics API example](#Metrics-YAML-File-Parsing-and-Metrics-API-Custom-Handler-Example)
* [Backwards compatibility warnings and upgrade guide](#backwards-compatibility-warnings-and-upgrade-guide)


## Introduction

Torchserve metrics can be broadly classified into frontend and backend metrics.

Frontend metrics include system level metrics. The host resource utilization frontend metrics are collected at regular intervals(default: every minute).

Torchserve provides [an API](#custom-metrics-api) to collect custom backend metrics. Metrics defined by a custom service or handler code can be collected per request or per a batch of requests.

Three metric modes are supported, i.e `log`, `prometheus` and `legacy` with the default mode being `log`.
The metrics mode can be configured using the `metrics_mode` configuration option in `config.properties` or `TS_METRICS_MODE` environment variable.
For further details on `config.properties` and environment variable based configuration, refer [Torchserve config](configuration.md) docs.

**Log Mode**

In `log` mode, metrics are logged and can be aggregated by metric agents.
Metrics are collected by default at the following locations in `log` mode:

* Frontend metrics - `log_directory/ts_metrics.log`
* Backend metrics - `log_directory/model_metrics.log`

The location of log files and metric files can be configured in the [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml) file

**Prometheus Mode**

In `prometheus` mode, metrics are made available in prometheus format via the [metrics API endpoint](metrics_api.md).

**Legacy Mode**

`legacy` mode enables backwards compatibility with Torchserve releases `<v0.8.0`, where:
* `ts_inference_requests_total`, `ts_inference_latency_microseconds` and `ts_queue_latency_microseconds` are only available via the [metrics API endpoint](metrics_api.md) in prometheus format.
* Frontend metrics are logged to `log_directory/ts_metrics.log`
* Backend metrics are logged to `log_directory/model_metrics.log`

`Note: To enable full backwards compatibility with releases <v0.8.0, use legacy metrics mode with `[model metrics auto-detection](#getting-started-with-torchserve-metrics)` enabled.`

## Getting Started with TorchServe Metrics

TorchServe defines metrics configuration in a [yaml](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml) file, including both frontend metrics (i.e. `ts_metrics`) and backend metrics (i.e. `model_metrics`).
When TorchServe is started, the metrics definition is loaded in the frontend and backend cache separately.
The backend emits metrics logs as they are updated. The frontend parses these logs and makes the corresponding metrics available either as logs or via the [metrics API endpoint](metrics_api.md) based on the `metrics_mode` configuration.


Dynamic updates to the metrics configuration file is not supported. In order to account for updates made to the metrics configuration file, Torchserve will need to be restarted.

By default, metrics that are not defined in the metrics configuration file will not be logged in the metrics log files or made available via the prometheus metrics API endpoint.
Backend model metrics can be `auto-detected` and registered in the frontend by setting `model_metrics_auto_detect` to `true` in `config.properties`
or using the `TS_MODEL_METRICS_AUTO_DETECT` environment variable. By default, `model_metrics_auto_detect` is disabled.

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

Metrics collected include:

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


### Metric Types Enum

TorchServe Metrics use [Metric Types](https://github.com/pytorch/serve/blob/master/ts/metrics/metric_type_enum.py)
that are in line with the [Prometheus API](https://github.com/prometheus/client_python) metric types.

Metric types are an attribute of Metric objects.
Users will be restricted to the existing metric types when adding metrics via Metrics API.

```python
class MetricTypes(enum.Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
```

## Metrics Formatting

TorchServe emits metrics to log files by default. The metrics are formatted in a [StatsD](https://github.com/etsy/statsd) like format.

```bash
CPUUtilization.Percent:0.0|#Level:Host|#hostname:my_machine_name,timestamp:1682098185
DiskAvailable.Gigabytes:318.0416717529297|#Level:Host|#hostname:my_machine_name,timestamp:1682098185
```

To enable metric logging in JSON format, set "patternlayout" as "JSONPatternLayout" in [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml) (See sample [log4j2-json.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/log4j2-json.xml)). For information, see [Logging in Torchserve](https://github.com/pytorch/serve/blob/master/docs/logging.md).

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

To enable metric logging in QLog format, set "patternlayout" as "QLogLayout" in [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml) (See sample [log4j2-qlog.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/log4j2-qlog.xml)). For information, see [Logging in Torchserve](https://github.com/pytorch/serve/blob/master/docs/logging.md).

After you enable QLogsetupModelDependencies formatting, logs will look as follows:

```qlog
HostName=abc.com
StartTime=1646686978
Program=MXNetModelServer
Metrics=MemoryUsed=5790.98046875 Megabytes Level|Host
EOE
HostName=147dda19895c.ant.amazon.com
StartTime=1646686978
Program=MXNetModelServer
Metrics=MemoryUtilization=46.2 Percent Level|Host
EOE

```

## Custom Metrics API

This is the API used in the backend handler to emit metrics. TorchServe enables the custom service code to emit metrics that are then made available based on the configured `metrics_mode`.

The custom service code is provided with a [context](https://github.com/pytorch/serve/blob/master/ts/context.py) of the current request with a metrics object:


```python
# Access context metrics as follows
metrics = context.metrics
```

All metrics are collected within the context.

**Note** The custom metrics API is not to be confused with the [metrics API endpoint](metrics_api.md) which is a http API that is used to fetch metrics in the prometheus format.

### Specifying Metric Types

When adding any metric via Metrics API, users have the ability to override the default metric type by specifying the positional argument
`metric_type=MetricTypes.[COUNTER/GAUGE/HISTOGRAM]`.

```python
example_metric = metrics.add_metric_to_cache(name="ExampleMetric", unit="ms", dimension_names=["name1", "name2"], metric_type=MetricTypes.GAUGE)
example_metric.add_or_update(value=1, dimension_values=["value1", "value2"])

# Backwards compatible, combines the above two method calls
metrics.add_metric(name="ExampleMetric", value=1, unit="ms", dimensions=[Dimension("name1", "value1"), Dimension("name2", "value2")], metric_type=MetricTypes.GAUGE)
```


### Updating Metrics parsed from the yaml file
Given the Metrics API, users will also be able to update metrics that have been parsed from the [yaml](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml) file
given some criteria:

(we will use the following metric as an example)
```yaml
  counter:  # metric type
    - name: InferenceTimeInMS  # name of metric
      unit: ms  # unit of metric
      dimensions: [ModelName, Level]
```
1. Metric Type has to be the same
   1. The user will have to use a counter-based `add_...` method,
      or explicitly set `metric_type=MetricTypes.counter` within the `add_...` method


2. Metric Name has to be the same
   1. If the name of the metric in the YAML file you want to update is `InferenceTimeInMS`, then `add_metric(name="InferenceTimeInMS", ...)`


3. Dimensions should be the same (as well as the same order!)
   1. All dimensions have to match, and Metric objects that have been parsed from the yaml file also have dimension names that are parsed from the yaml file
      1. Users can [create their own](#create-dimension-objects) `Dimension` objects to match those in the yaml file dimensions
      2. If the Metric object has `ModelName` and `Level` dimensions only, it is optional to specify additional dimensions since these are considered [default dimensions](#default-dimensions), so: `add_counter('InferenceTimeInMS', value=2)` or `add_counter('InferenceTimeInMS', value=2, dimensions=["ModelName", "Level"])`


### Default dimensions

Metrics will have a couple of default dimensions if not already specified:
  * `ModelName,{name_of_model}`
  * `Level,Model`


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

**Generic metrics are defaulted to a `COUNTER` metric type**

One can add metrics with generic units using the following function.

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
            list of dimension values
        request_id : str
            request id to be associated with the metric
        """
```

```python
# Add Distance as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size is 1 for example
metric = metrics.add_metric_to_cache('DistanceInKM', unit='km', dimension_names=[...])
metric.add_or_update(distance, dimension_values=[...])
```

Note that calling `add_metric_to_cache` will not emit the metric, `add_or_update` will need to be called on the metric object as shown above.

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
# Add Distance as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
metric = metrics.add_metric('DistanceInKM', value=10, unit='km', dimensions=[...])
```

### Add time-based metrics

**Time-based metrics are defaulted to a `GAUGE` metric type**

Add time-based by invoking the following method:

Function API

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
            list of dimensions for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Time metrics
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

**Size-based metrics are defaulted to a `GAUGE` metric type**

Add size-based metrics by invoking the following method:

Function API

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
            list of dimensions for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Size metrics
        """
```

Note that the default unit in this case is milliseconds (ms).

**Supported units**: `['MB', 'kB', 'GB', 'B']`

To add custom size based metrics

```python
# Add Image size as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size 1
metrics.add_size('SizeOfImage', img_size, None, 'MB', dimensions)
```

### Add Percentage based metrics

**Percentage-based metrics are defaulted to a `GAUGE` metric type**

Percentage based metrics can be added by invoking the following method:

Function API

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
            list of dimensions for the metric
        metric_type: MetricTypes
           type for defining different operations, defaulted to gauge metric type for Percent metrics
        """
```

**Inferred unit**: `percent`

To add custom percentage-based metrics:

```python
# Add MemoryUtilization as a metric
# dimensions = [dim1, dim2, dim3, ..., dimN]
# Assuming batch size 1
metrics.add_percent('MemoryUtilization', utilization_percent, None, dimensions)
```

### Add counter-based metrics

**Counter-based metrics are defaulted to a `COUNTER` metric type**

Counter based metrics can be added by invoking the following method

Function API

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
            list of dimensions for the metric
        """
```

**Inferred unit**: `count`

### Getting a metric

Users can get a metric from the cache. The CachingMetric object is returned, so the user can access the methods of the CachingMetric: (i.e. `CachingMetric.add_or_update(value, dimensions_values)`, `CachingMetric.update(value, dimensions)`)

```python
    def get_metric(self, metric_name: str, metric_type: MetricTypes) -> Metric:
        """
        Get a Metric from cache.
            Ask user for required requirements to form metric key to retrieve Metric.

        Parameters
        ----------
        metric_type: MetricTypes
            Type of metric: use MetricTypes enum to specify

        metric_name: str
            Name of metric

        """
```

i.e.
```python
# Method 1: Getting metric of metric name string, MetricType COUNTER
metrics.get_metric("MetricName", MetricTypes.COUNTER)

# Method 2: Getting metric of metric name string, MetricType GAUGE
metrics.get_metric("GaugeMetricName", MetricTypes.GAUGE)
```


## Emitting custom metrics

Following sample code can be used to emit the custom metrics created in the model's custom handler:

```python
# In Custom Handler
from ts.service import emit_metrics

class ExampleCustomHandler(BaseHandler, ABC):
   def initialize(self, ctx):

context.metrics.add_counter(...)
```

This custom metrics information is logged in the `model_metrics.log` file configured through [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml) file
or made available via the [metrics](metrics_api.md) API endpoint based on the `metrics_mode` configuration.

## Metrics YAML File Parsing and Metrics API Custom Handler Example

This example utilizes the feature of parsing metrics from a YAML file, adding and updating metrics and their values via Metrics API,
updating metrics that have been parsed from the YAML file via Metrics API, and finally emitting all metrics that have been updated.

```python
from ts.service import emit_metrics
from ts.metrics.metric_type_enum import MetricTypes


class CustomHandlerExample:
    def initialize(self, ctx):
        metrics = ctx.metrics  # initializing metrics to the context.metrics

        # Setting a sleep for examples' sake
        start_time = time.time()
        time.sleep(3)
        stop_time = time.time()

        # Adds a metric that has a metric type of gauge
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )

        # Logs the value 2.5 and -1.3 to the frontend
        metrics.add_counter("HandlerSeparateCounter", 2.5)
        metrics.add_counter("HandlerSeparateCounter", -1.3)

        # Adding a standard counter metric
        metrics.add_counter("HandlerCounter", 21.3)

        # Assume that a metric that has a metric type of counter
        # and is named InferenceTimeInMS in the metrics.yaml file.
        # Instead of creating a new object with the same name and same parameters,
        # this line will update the metric that already exists from the YAML file.
        metrics.add_counter("InferenceTimeInMS", 2.78)

        # Another method of updating values -
        # using the get_metric + Metric.update method
        # In this example, we are getting an already existing
        # Metric that had been parsed from the yaml file
        histogram_example_metric = metrics.get_metric(
            "HistogramModelMetricNameExample",
            MetricTypes.histogram,
        )
        histogram_example_metric.add_or_update(4.6)

        # Same idea as the 'metrics.add_counter('InferenceTimeInMS', 2.78)' line,
        # except this time with gauge metric type object
        metrics.add_size("GaugeModelMetricNameExample", 42.5)
```

## Backwards compatibility warnings and upgrade guide
1. Starting [v0.6.1](https://github.com/pytorch/serve/releases/tag/v0.6.1), the `add_metric` API signature changed\
   from: [add_metric(name, value, unit, idx=None, dimensions=None)](https://github.com/pytorch/serve/blob/61f1c4182e6e864c9ef1af99439854af3409d325/ts/metrics/metrics_store.py#L184)\
   to: [add_metric(metric_name, unit, dimension_names=None, metric_type=MetricTypes.COUNTER)](https://github.com/pytorch/serve/blob/35ef00f9e62bb7fcec9cec92630ae757f9fb0db0/ts/metrics/metric_cache_abstract.py#L272).\
   In versions greater than v0.8.1 the `add_metric` API signature was updated to support backwards compatibility:\
   from: [add_metric(metric_name, unit, dimension_names=None, metric_type=MetricTypes.COUNTER)](https://github.com/pytorch/serve/blob/35ef00f9e62bb7fcec9cec92630ae757f9fb0db0/ts/metrics/metric_cache_abstract.py#L272)\
   to: `add_metric(name, value, unit, idx=None, dimensions=[], metric_type=MetricTypes.COUNTER)`\
   Usage of the new API is shown [above](#specifying-metric-types).
   **Upgrade paths**:
   - **[< v0.6.1] to [v0.6.1 - v0.8.1]**\
   There are two approaches available when migrating to the new custom metrics API:
     - Replace the call to `add_metric` with calls to the following methods:
       ```python
       metric1 = metrics.add_metric("GenericMetric", unit=unit, dimension_names=["name1", "name2", ...], metric_type=MetricTypes.GAUGE)
       metric1.add_or_update(value, dimension_values=["value1", "value2", ...])
       ```
     - Replace the call to `add_metric` in versions prior to v0.6.1 with one of the suitable custom metrics APIs where applicable: [add_counter](#add-counter-based-metrics), [add_time](#add-time-based-metrics),
       [add_size](#add-size-based-metrics) or [add_percent](#add-percentage-based-metrics)
   - **[< v0.6.1] to [> v0.8.1]**\
     The call to `add_metric` is backwards compatible but the metric type is inferred to be `COUNTER`. If the metric is of a different type, an additional argument `metric_type` will need to be provided to the `add_metric`
     call shown below
     ```python
     metrics.add_metric(name='GenericMetric', value=10, unit='count', dimensions=[...], metric_type=MetricTypes.GAUGE)
     ```
   - **[v0.6.1 - v0.8.1] to [> v0.8.1]**\
     Replace the call to `add_metric` with `add_metric_to_cache`.
2. In versions [[v0.8.0](https://github.com/pytorch/serve/releases/tag/v0.8.0) - [v0.9.0](https://github.com/pytorch/serve/releases/tag/v0.9.0)], only metrics that are defined in the metrics config file(default: [metrics.yaml](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml))
   are either all logged to `ts_metrics.log` and `model_metrics.log` or made available via the [metrics API endpoint](metrics_api.md)
   based on the `metrics_mode` configuration as described [above](#introduction).\
   The default `metrics_mode` is `log` mode.\
   This is unlike in previous versions where all metrics were only logged to `ts_metrics.log` and `model_metrics.log` except for `ts_inference_requests_total`, `ts_inference_latency_microseconds` and `ts_queue_latency_microseconds`
   which were only available via the metrics API endpoint.\
   **Upgrade paths**:
   - **[< v0.8.0] to [v0.8.0 - v0.9.0]**\
     Specify all the custom metrics added to the custom handler in the metrics configuration file as shown [above](#getting-started-with-torchserve-metrics).
   - **[< v0.8.0] to [> v0.9.0]**\
     Set `metrics_mode` to `legacy` and enable [model metrics auto-detection](#getting-started-with-torchserve-metrics).
