package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Counter;
import org.pytorch.serve.metrics.IMetrics;

import java.util.ArrayList;

public class PrometheusCounter extends IMetrics {
    public PrometheusCounter(String metricsType, String metricsName, String unit, ArrayList<String> dimensionNames) {
        super(metricsType, metricsName, unit, dimensionNames);
    }

    @Override
    public void emit(ArrayList<String> dimensionValues, double value) {

    }

    @Override
    public void emit(ArrayList<String> dimensionValues, String requestIds, double value) {

    }

    private Counter counter;
}
