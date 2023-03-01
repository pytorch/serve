package org.pytorch.serve.metrics;

import java.util.ArrayList;

public class LogMetric extends IMetric {
    public LogMetric(
            MetricBuilder.MetricType metricsType,
            String metricsName,
            String unit,
            ArrayList<String> dimensionNames) {
        super(metricsType, metricsName, unit, dimensionNames);
    }

    @Override
    public void emit(ArrayList<String> dimensionValues, double value) {

    }

    @Override
    public void emit(ArrayList<String> dimensionValues, String requestIds, double value) {

    }
}
