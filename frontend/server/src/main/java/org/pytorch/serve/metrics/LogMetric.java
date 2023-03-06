package org.pytorch.serve.metrics;

import java.util.ArrayList;

public class LogMetric extends IMetric {
    /**
     * Note:
     * hostname, timestamp, and requestid(if available) are automatically added in log metric.
     */
    public LogMetric(
            MetricBuilder.MetricType metricsType,
            String metricsName,
            String unit,
            ArrayList<String> dimensionNames) {
        super(metricsType, metricsName, unit, dimensionNames);
    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, double value) {
    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, String requestIds, double value) {

    }
}
