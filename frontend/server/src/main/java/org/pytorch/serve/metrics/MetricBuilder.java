package org.pytorch.serve.metrics;

import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;

import java.util.ArrayList;

public class MetricBuilder {
    public static final IMetric build(
            MetricMode mode,
            MetricType metricsType,
            String metricsName,
            String unit,
            ArrayList<String> dimensionNames) {
        if (mode == MetricMode.PROMETHEUS) {
            if (metricsType == MetricType.COUNTER) {
                return new PrometheusCounter(metricsType, metricsName, unit, dimensionNames);
            } else if (metricsType == MetricType.GAUGE) {
                return new PrometheusGauge(metricsType, metricsName, unit, dimensionNames);
            }
        } else {
            return new LogMetric(metricsType, metricsName, unit, dimensionNames);
        }
        return null;
    }

    public enum MetricMode {
        PROMETHEUS,
        LOG
    }

    public enum MetricType {
        COUNTER,
        GAUGE
    }
}
