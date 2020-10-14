package org.pytorch.serve.test.plugins.metrics;

import java.util.List;
import org.pytorch.serve.servingsdk.metrics.BaseDimension;
import org.pytorch.serve.servingsdk.metrics.BaseMetric;
import org.pytorch.serve.servingsdk.metrics.DimensionRegistry;
import org.pytorch.serve.servingsdk.metrics.InbuiltMetricsRegistry;
import org.pytorch.serve.servingsdk.metrics.MetricEventListener;
import org.pytorch.serve.servingsdk.metrics.MetricLogEvent;


public class TestMetricEventListener implements MetricEventListener {

    @Override
    public void handle(MetricLogEvent metricLogEvent) {
        BaseMetric metric = metricLogEvent.getMetric();
        String metricName = metric.getMetricName();
        List<BaseDimension> dimensions = metric.getDimensions();
        String dimLevel= null;
        String dimModelName = null;
        String dimModelVersion = null;
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
            default:
                break;
        }

    }
}