package org.pytorch.serve.servingsdk.metrics;

import java.util.Date;

/**
 * This interface specifies add and remove listener methods
 */
public interface MetricLogEvent {
    /**
     * Get the log level
     * @return The log level
     */
    String getLevel();

    /**
     * Get the logged message as a  BaseMetric
     * @return BaseMetric object
     */
    BaseMetric getMetric();

    /**
     * Get the log Timestamp
     * @return The Timestamp
     */
    Date getTimestamp();
}