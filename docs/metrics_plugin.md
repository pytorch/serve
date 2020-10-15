# Add or extend Metrics plugin

This document explains the Prometheus Metrics plugin, which will help users to extend or add new Metrics plugin.

## General overview of Plugin functionality
We have three components in TorchServe - TorchServe Server, [SDK](../serving-sdk) and [Plugins](../plugins).
The SDK provides contracts in the form of interfaces, abstract classes. The SDK jar is already added as a dependency
in build.gradle for TorchServe Server and plugin modules. The Server and Plugin implements all the necessary interfaces
from SDK.

## Understanding Prometheus Metrics Endpoint plugin
The TorchServe Server emits different metrics as Metric logger statements. The emitted logger statements
are intercepted by the MetricEventPublisher and are then broadcasted to the listeners. The MetricEventListenerRegistry
helps in adding new listeners to the publisher. The Prometheus plugin creates a MetricEventListener and registers itself
with the registry. Whenever a metric event is received by the listener, it adds it to the Prometheus local metric registry,
which is used by the Metric endpoint to serve the GET metrics requests.

Below are three interfaces or abstract class contracts implemented by the Prometheus plugin.

1. [PrometheusMetrics extends ModelServerEndpoint](../plugins/prometheus_metric_endpoint/src/main/java/org/pytorch/serve/plugins/endpoint/prometheus/PrometheusMetrics.java)
1. [PrometheusMetricEventListenerRegistry extends MetricEventListenerRegistry](../plugins/prometheus_metric_endpoint/src/main/java/org/pytorch/serve/plugins/endpoint/prometheus/PrometheusMetricEventListenerRegistry.java)
1. [PrometheusMetricEventListener implements MetricEventListener](../plugins/prometheus_metric_endpoint/src/main/java/org/pytorch/serve/plugins/endpoint/prometheus/MetricEventListener.java)

Below is the detailed explanation for the three:
1. [PrometheusMetrics extends ModelServerEndpoint](../plugins/prometheus_metric_endpoint/src/main/java/org/pytorch/serve/plugins/endpoint/prometheus/PrometheusMetrics.java)  
    This class registers a API of type Metrics with 'metrics' as a url path. The get method is implemented. It accepts metric filter
    path parameters and provides metrics collected in Prometheus registry.
    The PrometheusMetrics endpoint get registered with TorchServe in the server initialization phase. The [service loader](plugins/prometheus_metric_endpoint/src/main/resources/META-INF/services/org.pytorch.serve.servingsdk.ModelServerEndpoint)
    is used by TorchServe to load the PrometheusMetrics endpoint class.
2. [PrometheusMetricEventListenerRegistry extends MetricEventListenerRegistry](../plugins/prometheus_metric_endpoint/src/main/java/org/pytorch/serve/plugins/endpoint/prometheus/PrometheusMetricEventListenerRegistry.java)  
    This class is used to register PrometheusMetricEventListener with the MetricEventPublisher.
    The class gets loaded in the server initialization phase and MetricEventPublisher is passed in for registration.
3. [PrometheusMetricEventListener implements MetricEventListener](../plugins/prometheus_metric_endpoint/src/main/java/org/pytorch/serve/plugins/endpoint/prometheus/MetricEventListener.java)  
    This class handles the MetricLogEvent broadcasted by the publisher. The metric name, value and dimensions are captured
    from MetricLogEvent and metrics are published to Prometheus registry.
    Whenever there is new Metric logged by metric loggers [TS_METRICS](frontend/server/src/main/resources/log4j.properties), [MODEL_METRICS](frontend/server/src/main/resources/log4j.properties), the event get broadcasted to listeners.

 Feel free to go through the codebase to better understand the implementation. You can extend Prometheus plugin as per your needs or add a new one.

 ## What are the metrics supported?
 Please refer the [document](metrics.md) to check different metrics supported by TorchServe. If you want to add your own metric it can be added as well.
 
 ## How to build and use the plugin?
 Please refer [document](metrics_api.md) for details.