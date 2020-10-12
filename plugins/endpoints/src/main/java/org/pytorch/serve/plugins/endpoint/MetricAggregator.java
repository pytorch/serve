package org.pytorch.serve.plugins.endpoint;

import org.pytorch.serve.plugins.endpoint.PrometheusMetricManager;
import org.pytorch.serve.servingsdk.Context;


public final class MetricAggregator {

    private MetricAggregator() {}

    public static void handleInferenceMetric( final String modelName, final String modelVersion) {
        PrometheusMetricManager.getInstance().incInferCount(modelName, modelVersion);
    }

    public static void handleInferenceMetric(
            final String modelName, final String modelVersion, long timeInQueue, long inferTime) {

        PrometheusMetricManager metrics = PrometheusMetricManager.getInstance();
        metrics.incInferLatency(inferTime, modelName, modelVersion);
        metrics.incQueueLatency(timeInQueue, modelName, modelVersion);

    }
}


