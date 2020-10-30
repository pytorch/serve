package org.pytorch.serve.servingsdk.metrics;

/**
 * This interface specifies method to add and remove listener.
 *
 */
public interface MetricEventPublisher {
    void addMetricEventListener(MetricEventListener listener);
    void removeMetricEventListener(MetricEventListener listener);
}


