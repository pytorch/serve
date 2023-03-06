package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Gauge;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricBuilder;

import java.util.ArrayList;

public class PrometheusGauge extends IMetric {
    public PrometheusGauge(
            MetricBuilder.MetricType metricsType,
            String metricsName,
            String unit,
            ArrayList<String> dimensionNames) {
        super(metricsType, metricsName, unit, dimensionNames);
    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, double value) {

    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, String requestIds, double value) {

    }

    private Gauge gauge;
}
