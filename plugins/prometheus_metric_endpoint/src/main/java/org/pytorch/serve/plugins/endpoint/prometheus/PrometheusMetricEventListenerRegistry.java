package org.pytorch.serve.plugins.endpoint.prometheus;

import org.pytorch.serve.servingsdk.metrics.*;
import java.util.List;

public class PrometheusMetricEventListenerRegistry extends MetricEventListenerRegistry {
    public void register(MetricEventPublisher publisher) {
        PrometheusMetricEventListener listener = new PrometheusMetricEventListener();
        publisher.addMetricEventListener(listener);
    }
}

class PrometheusMetricEventListener implements MetricEventListener {

    @Override
    public void handle(MetricLogEvent metricLogEvent) {
        PrometheusMetricManager prometheusMetricManager = PrometheusMetricManager.getInstance();
        BaseMetric metric = metricLogEvent.getMetric();
        String metricName = metric.getMetricName();
        List<BaseDimension> dimensions = metric.getDimensions();
        String dimLevel= null, dimModelName = null, dimModelVersion = null;
        for (BaseDimension dimension : dimensions) {
            switch (dimension.getName()) {
                case DimensionRegistry.LEVEL:
                    dimLevel = dimension.getValue();  break;
                case DimensionRegistry.MODELNAME:
                    dimModelName =  dimension.getValue();  break;
                case DimensionRegistry.MODELVERSION:
                    dimModelVersion =  dimension.getValue();  break;
            }
        }

        switch (metricName) {
            case (InbuiltMetricsRegistry.INFERENCE):
                for (int i = 1; i <= Integer.parseInt(metric.getValue()); i++) {
                    prometheusMetricManager.incInferCount(dimModelName, dimModelVersion);
                }
                break;
            case (InbuiltMetricsRegistry.QUEUETIME):
                prometheusMetricManager.incQueueLatency(Double.parseDouble(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case (InbuiltMetricsRegistry.BACKENDRESPONSETIME):
                prometheusMetricManager.incBackendResponseLatency(Double.parseDouble(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case (InbuiltMetricsRegistry.HANDLERTIME):
                prometheusMetricManager.incHandlerLatency(Double.parseDouble(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case ("MemoryUsed"):
                prometheusMetricManager.addMemoryUsed(Double.parseDouble(metric.getValue()));
                break;
        }

    }
}