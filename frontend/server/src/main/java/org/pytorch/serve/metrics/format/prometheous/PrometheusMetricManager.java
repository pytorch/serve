package org.pytorch.serve.metrics.format.prometheous;

import io.prometheus.client.Counter;
import java.util.UUID;

public final class PrometheusMetricManager {

    private static final PrometheusMetricManager METRIC_MANAGER = new PrometheusMetricManager();
    private static final String METRICS_UUID = UUID.randomUUID().toString();
    private Counter inferRequestCount;
    private Counter inferLatency;
    private Counter queueLatency;

    private PrometheusMetricManager() {
        String[] metricsLabels = {"uuid", "model_name", "model_version"};
        inferRequestCount =
                Counter.build()
                        .name("ts_inference_requests_total")
                        .labelNames(metricsLabels)
                        .help("Total number of inference requests.")
                        .register();
        inferLatency =
                Counter.build()
                        .name("ts_inference_latency_microseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative inference duration in microseconds")
                        .register();
        queueLatency =
                Counter.build()
                        .name("ts_queue_latency_microseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative queue duration in microseconds")
                        .register();
    }

    private static String getOrDefaultModelVersion(String modelVersion) {
        return modelVersion == null ? "default" : modelVersion;
    }

    public static PrometheusMetricManager getInstance() {
        return METRIC_MANAGER;
    }

    /**
     * Counts the time in ns it took for an inference to be completed
     *
     * @param inferTime time in nanoseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incInferLatency(long inferTime, String modelName, String modelVersion) {
        inferLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(inferTime / 1000.0);
    }

    /**
     * Counts the time in ns an inference request was queued before being executed
     *
     * @param queueTime time in nanoseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incQueueLatency(long queueTime, String modelName, String modelVersion) {
        queueLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(queueTime / 1000.0);
    }

    /**
     * Counts a valid inference request to be processed
     *
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incInferCount(String modelName, String modelVersion) {
        inferRequestCount
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc();
    }
}
