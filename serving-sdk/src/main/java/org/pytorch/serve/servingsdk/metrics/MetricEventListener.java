package org.pytorch.serve.servingsdk.metrics;

/**
 * This is a listener which listens to MetricLogEvent
 */

public interface MetricEventListener {
    /**
     * Handle the MetricLogEvent
     * It gets called when MetricEventPublisher broadcasts the event to listeners
     * @param event - MetricLogEvent
     */
    void handle(MetricLogEvent event);
}

