package org.pytorch.serve.metrics;

import java.util.List;
import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;
import org.pytorch.serve.metrics.format.prometheous.PrometheusHistogram;

public final class MetricBuilder {
    public enum MetricMode {
        PROMETHEUS,
        LOG
    }

    public enum MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM
    }

    public static final IMetric build(
            MetricMode mode,
            MetricType type,
            String name,
            String unit,
            List<String> dimensionNames) {
        if (mode == MetricMode.PROMETHEUS) {
            switch (type) {
                case COUNTER:
                    return new PrometheusCounter(type, name, unit, dimensionNames);
                case GAUGE:
                    return new PrometheusGauge(type, name, unit, dimensionNames);
                case HISTOGRAM:
                    return new PrometheusHistogram(type, name, unit, dimensionNames);
                default:
            }
        } else {
            return new LogMetric(type, name, unit, dimensionNames);
        }
        return null;
    }

    private MetricBuilder() {}
}
