package org.pytorch.serve.servingsdk.metrics;

/**
 * This provides information about the model which is currently registered with Model Server
 */

public interface MetricEventListener {
    /**
     * Handle the LogEvent
     */
    void handle(MetricLogEvent event);
}

