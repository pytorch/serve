package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Histogram;
import java.util.List;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricBuilder;

public class PrometheusHistogram extends IMetric {
    private final Histogram histogram;

    public PrometheusHistogram(
            MetricBuilder.MetricType type, String name, String unit, List<String> dimensionNames) {
        super(type, name, unit, dimensionNames);
        this.histogram =
                Histogram.build()
                        .name(this.name)
                        .labelNames(
                                this.dimensionNames.toArray(new String[this.dimensionNames.size()]))
                        .help("Torchserve prometheus histogram metric with unit: " + this.unit)
                        .register();
    }

    @Override
    public void addOrUpdate(List<String> dimensionValues, double value) {
        this.histogram
                .labels(dimensionValues.toArray(new String[dimensionValues.size()]))
                .observe(value);
    }

    @Override
    public void addOrUpdate(List<String> dimensionValues, String requestIds, double value) {
        this.addOrUpdate(dimensionValues, value);
    }
}
