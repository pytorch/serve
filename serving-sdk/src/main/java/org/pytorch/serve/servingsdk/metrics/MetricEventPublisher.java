package org.pytorch.serve.servingsdk.metrics;


public interface MetricEventPublisher {
    void addMetricEventListener(MetricEventListener listener);
    void removeMetricEventListener(MetricEventListener listener);
}


