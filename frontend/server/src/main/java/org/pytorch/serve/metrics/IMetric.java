package org.pytorch.serve.metrics;

import java.util.ArrayList;

public abstract class IMetric {
    public IMetric(
            MetricBuilder.MetricType metricsType,
            String metricsName,
            String unit,
            ArrayList<String> dimensionNames) {
        this.metricsType = metricsType;
        this.metricsName = metricsName;
        this.unit = unit;
        this.dimensionNames = dimensionNames;
    }

    public abstract void emit(
            ArrayList<String> dimensionValues,
            double value);
    public abstract void emit(
            ArrayList<String> dimensionValues,
            String requestIds,
            double value);

    private MetricBuilder.MetricType metricsType;
    private String metricsName;
    private String unit;
    private ArrayList<String> dimensionNames;
}
