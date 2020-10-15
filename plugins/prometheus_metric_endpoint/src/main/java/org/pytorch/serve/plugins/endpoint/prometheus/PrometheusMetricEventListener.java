package org.pytorch.serve.plugins.endpoint.prometheus;

import java.util.List;
import org.pytorch.serve.servingsdk.metrics.*;

/** It listens for the Metric log events and updates the Prometheus metric values */
public class PrometheusMetricEventListener implements MetricEventListener {
    /**
     * Listens for MetricLogEvent and updates the Prometheus metrics
     *
     * @param metricLogEvent MetricLogEvent object
     */
    @Override
    public void handle(MetricLogEvent metricLogEvent) {
        PrometheusMetricManager prometheusMetricManager = PrometheusMetricManager.getInstance();
        BaseMetric metric = metricLogEvent.getMetric();
        String metricName = metric.getMetricName();
        List<BaseDimension> dimensions = metric.getDimensions();

        // Get the dimensions at which the metric values are captured
        // DimensionRegistry lists the different dimensions for reference
        // However note that its not exhaustive list
        String dimModelName = null, dimModelVersion = null;
        for (BaseDimension dimension : dimensions) {
            switch (dimension.getName()) {
                case DimensionRegistry.MODELNAME:
                    dimModelName = dimension.getValue();
                    break;
                case DimensionRegistry.MODELVERSION:
                    dimModelVersion = dimension.getValue();
                    break;
            }
        }

        // Get the dimensions at which the values are captured
        // InbuiltMetricsRegistry lists the different metrics emitted by TorchServe
        // However note that its not exhaustive list, for.ex. custom metrics are not part of this
        // list
        switch (metricName) {
            case (InbuiltMetricsRegistry.INFERENCEREQUESTS):
                prometheusMetricManager.incInferCount(
                        Integer.parseInt(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case (InbuiltMetricsRegistry.QUEUETIME):
                prometheusMetricManager.incQueueLatency(
                        Double.parseDouble(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case (InbuiltMetricsRegistry.BACKENDRESPONSETIME):
                prometheusMetricManager.incBackendResponseLatency(
                        Double.parseDouble(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case (InbuiltMetricsRegistry.HANDLERTIME):
                prometheusMetricManager.incHandlerLatency(
                        Double.parseDouble(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case ("MemoryUsed"): // Example of System metric.
                prometheusMetricManager.addMemoryUsed(Double.parseDouble(metric.getValue()));
                break;
        }
    }
}
