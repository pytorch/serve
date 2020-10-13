package org.pytorch.serve.servingsdk.metrics;

/**
 * This provides information about the Metric Dimension
 */
public interface BaseDimension {
    /**
     * @return Metric Dimension Name
     */
    String getName();

    /**
     * @return Metric Dimension Value
     */
    String getValue();
}