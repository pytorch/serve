package org.pytorch.serve.servingsdk.metrics;

/**
 * This is a listener which listens to MetricLogEvent
 */

public interface MetricEventListener {
    /**
     * Handle the MetricLogEvent
     * @param event - MetricLogEvent
     */
    void handle(MetricLogEvent event);
}

