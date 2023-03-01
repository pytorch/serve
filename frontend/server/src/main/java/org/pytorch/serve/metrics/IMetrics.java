package org.pytorch.serve.metrics;

import java.util.ArrayList;

public abstract class IMetrics {
    public IMetrics(
            String metricsType,
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

    private String metricsType;
    private String metricsName;
    private String unit;
    private ArrayList<String> dimensionNames;
}
