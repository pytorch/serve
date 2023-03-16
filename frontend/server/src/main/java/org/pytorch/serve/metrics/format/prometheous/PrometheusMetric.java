package org.pytorch.serve.metrics.format.prometheous;

import java.util.ArrayList;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricBuilder;

public abstract class PrometheusMetric extends IMetric {
    public PrometheusMetric(
            MetricBuilder.MetricContext context,
            MetricBuilder.MetricType type,
            String name,
            String unit,
            ArrayList<String> dimensionNames) {
        super(context, type, name, unit, dimensionNames);

        if (this.context == MetricBuilder.MetricContext.BACKEND) {
            this.dimensionNames.add("Hostname");
        }
    }

    @Override
    public void addOrUpdate(
            ArrayList<String> dimensionValues,
            String hostname,
            String requestIds,
            String timestamp,
            double value) {
        ArrayList<String> modifiedDimensionValues = new ArrayList<String>(dimensionValues);
        if (this.context == MetricBuilder.MetricContext.BACKEND
                && hostname != null
                && !hostname.isEmpty()) {
            modifiedDimensionValues.add(hostname);
        }

        this.addOrUpdate(modifiedDimensionValues, value);
    }
}
