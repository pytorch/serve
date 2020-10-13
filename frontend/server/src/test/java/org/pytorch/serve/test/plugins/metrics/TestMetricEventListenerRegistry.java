package org.pytorch.serve.test.plugins.metrics;

import org.pytorch.serve.servingsdk.metrics.*;

import java.util.List;

public class TestMetricEventListenerRegistry extends MetricEventListenerRegistry {
    public void register(MetricEventPublisher publisher) {
        TestMetricEventListener listener = new TestMetricEventListener();
        publisher.addMetricEventListener(listener);
    }

}

class TestMetricEventListener implements MetricEventListener {

    @Override
    public void handle(MetricLogEvent metricLogEvent) {
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

        TestMetricManager metricManager = TestMetricManager.getInstance();

        switch (metricName) {
            case (InbuiltMetricsRegistry.INFERENCE):
                metricManager.incInferRequestCount(Integer.parseInt(metric.getValue()));
                break;
            case (InbuiltMetricsRegistry.QUEUETIME):
                metricManager.incQueueLatency(Double.parseDouble(metric.getValue()));
                break;
            case (InbuiltMetricsRegistry.BACKENDRESPONSETIME):
                metricManager.incBackendResponseLatency(Double.parseDouble(metric.getValue()));
                break;
            case (InbuiltMetricsRegistry.HANDLERTIME):
                metricManager.incHandlerlatency(Double.parseDouble(metric.getValue()));
                break;
            case ("MemoryUsed"):
                metricManager.incMemoryUsed(Double.parseDouble(metric.getValue()));
                break;
        }

    }
}