package org.pytorch.serve.plugins.endpoint;

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
                case "Level":
                    dimLevel = dimension.getValue();  break;
                case "ModelName":
                    dimModelName =  dimension.getValue();  break;
                case "ModelVersion":
                    dimModelVersion =  dimension.getValue();  break;
            }
        }

        switch (metricName.toUpperCase()) {
            case ("INFERENCE"):
                for (int i = 1; i <= Integer.parseInt(metric.getValue()); i++) {
                    prometheusMetricManager.incInferCount(dimModelName, dimModelVersion);
                }
                break;
            case ("QUEUETIME"):
                prometheusMetricManager.incQueueLatency(Long.parseLong(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case ("BACKENDRESPONSETIME"):
                prometheusMetricManager.incBackendResponseLatency(Long.parseLong(metric.getValue()), dimModelName, dimModelVersion);
                break;
            case ("HANDLERTIME"):
                prometheusMetricManager.incHandlerLatency(Long.parseLong(metric.getValue()), dimModelName, dimModelVersion);

        }

    }
}