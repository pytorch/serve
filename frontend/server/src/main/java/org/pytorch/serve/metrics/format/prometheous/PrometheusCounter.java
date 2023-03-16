package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Counter;
import java.util.ArrayList;
import org.pytorch.serve.metrics.MetricBuilder;

public class PrometheusCounter extends PrometheusMetric {
    private final Counter counter;

    public PrometheusCounter(
            MetricBuilder.MetricContext context,
            MetricBuilder.MetricType type,
            String name,
            String unit,
            ArrayList<String> dimensionNames) {
        super(context, type, name, unit, dimensionNames);
        this.counter =
                Counter.build()
                        .name(this.name)
                        .labelNames(this.dimensionNames.toArray(new String[0]))
                        .help("Torchserve metric with unit: " + this.unit)
                        .register();
    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, double value) {
        this.counter.labels(dimensionValues.toArray(new String[0])).inc(value);
    }
}
