package org.pytorch.serve.servingsdk.metrics;

import java.util.Date;


public interface MetricLogEvent {
    /**
     * Get the log level
     * @return The name of this model
     */
    String getLevel();

    /**
     * Get the name of this model
     * @return The name of this model
     */
    BaseMetric getMetric();

    /**
     * Get the name of this model
     * @return The name of this model
     */
    Date getTimestamp();
}



