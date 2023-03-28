package org.pytorch.serve.metrics;

import java.util.ArrayList;
import java.util.List;

public abstract class IMetric {
    protected MetricBuilder.MetricContext context;
    protected MetricBuilder.MetricType type;
    protected String name;
    protected String unit;
    protected List<String> dimensionNames;

    public IMetric(
            MetricBuilder.MetricContext context,
            MetricBuilder.MetricType type,
            String name,
            String unit,
            List<String> dimensionNames) {
        this.context = context;
        this.type = type;
        this.name = name;
        this.unit = unit;
        this.dimensionNames = new ArrayList<String>(dimensionNames);
    }

    public abstract void addOrUpdate(List<String> dimensionValues, double value);

    public abstract void addOrUpdate(
            List<String> dimensionValues, String hostname, String requestIds, double value);
}
