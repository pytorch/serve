package org.pytorch.serve.servingsdk.metrics;

/**
 * This interface specifies add and remove listener methods
 */
public interface MetricEventPublisher {
    void addMetricEventListener(MetricEventListener listener);
    void removeMetricEventListener(MetricEventListener listener);
}


