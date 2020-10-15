package org.pytorch.serve.servingsdk.metrics;

import java.util.List;

/**
 * This provides information about the Metric
 */
public interface BaseMetric {

    /**
     * @return Host Name
     */
    String getHostName();

    /**
     * @return request id
     */
    String getRequestId();

    /**
     * @return Metric Name
     */
    String getMetricName();

    /**
     * @return Metric Value
     */
    String getValue();

    /**
     * @return Metric value unit. Can be 'ms', 'Count', 'GB' etc.
     */
    String getUnit();

    /**
     * @return List of Metric Dimensions
     */
    List<BaseDimension> getDimensions();

    /**
     * @return timestamp
     */
    String getTimestamp();
}


