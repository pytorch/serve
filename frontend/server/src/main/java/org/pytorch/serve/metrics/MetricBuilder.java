package org.pytorch.serve.metrics;

import java.util.ArrayList;
import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;

public final class MetricBuilder {
    public enum MetricMode {
        PROMETHEUS,
        LOG
    }

    public enum MetricContext {
        FRONTEND,
        BACKEND
    }

    public enum MetricType {
        COUNTER,
        GAUGE
    }

    public static final IMetric build(
            MetricMode mode,
            MetricContext context,
            MetricType type,
            String name,
            String unit,
            ArrayList<String> dimensionNames) {
        if (mode == MetricMode.PROMETHEUS) {
            if (type == MetricType.COUNTER) {
                return new PrometheusCounter(context, type, name, unit, dimensionNames);
            } else if (type == MetricType.GAUGE) {
                return new PrometheusGauge(context, type, name, unit, dimensionNames);
            }
        } else {
            return new LogMetric(context, type, name, unit, dimensionNames);
        }
        return null;
    }

    private MetricBuilder() {
        throw new UnsupportedOperationException();
    }
}
