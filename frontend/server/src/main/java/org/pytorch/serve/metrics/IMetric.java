package org.pytorch.serve.metrics;

import java.util.ArrayList;

public abstract class IMetric {
    protected MetricBuilder.MetricContext context;
    protected MetricBuilder.MetricType type;
    protected String name;
    protected String unit;
    protected ArrayList<String> dimensionNames;

    public IMetric(
            MetricBuilder.MetricContext context,
            MetricBuilder.MetricType type,
            String name,
            String unit,
            ArrayList<String> dimensionNames) {
        this.context = context;
        this.type = type;
        this.name = name;
        this.unit = unit;
        this.dimensionNames = new ArrayList<String>(dimensionNames);
    }

    public abstract void addOrUpdate(
            ArrayList<String> dimensionValues,
            double value);

    public abstract void addOrUpdate(
            ArrayList<String> dimensionValues,
            String hostname,
            String requestIds,
            String timestamp,
            double value);
}
