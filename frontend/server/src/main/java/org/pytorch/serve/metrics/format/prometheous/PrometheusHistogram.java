package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Histogram;
import java.util.ArrayList;
import org.pytorch.serve.metrics.MetricBuilder;

public class PrometheusHistogram extends PrometheusMetric {
    private final Histogram histogram;

    public PrometheusHistogram(
            MetricBuilder.MetricContext context,
            MetricBuilder.MetricType type,
            String name,
            String unit,
            ArrayList<String> dimensionNames) {
        super(context, type, name, unit, dimensionNames);
        this.histogram =
                Histogram.build()
                        .name(this.name)
                        .labelNames(
                                this.dimensionNames.toArray(new String[this.dimensionNames.size()]))
                        .help("Torchserve prometheus histogram metric with unit: " + this.unit)
                        .register();
    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, double value) {
        this.histogram
                .labels(dimensionValues.toArray(new String[dimensionValues.size()]))
                .observe(value);
    }
}
