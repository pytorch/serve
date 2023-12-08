package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Counter;
import java.util.List;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricBuilder;

public class PrometheusCounter extends IMetric {
    private final Counter counter;

    public PrometheusCounter(
            MetricBuilder.MetricType type, String name, String unit, List<String> dimensionNames) {
        super(type, name, unit, dimensionNames);
        this.counter =
                Counter.build()
                        .name(this.name)
                        .labelNames(
                                this.dimensionNames.toArray(new String[this.dimensionNames.size()]))
                        .help("Torchserve prometheus counter metric with unit: " + this.unit)
                        .register();
    }

    @Override
    public void addOrUpdate(List<String> dimensionValues, double value) {
        this.counter.labels(dimensionValues.toArray(new String[dimensionValues.size()])).inc(value);
    }

    @Override
    public void addOrUpdate(List<String> dimensionValues, String requestIds, double value) {
        this.addOrUpdate(dimensionValues, value);
    }
}
