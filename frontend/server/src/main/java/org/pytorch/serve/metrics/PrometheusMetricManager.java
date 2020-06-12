package org.pytorch.serve.metrics;

import io.prometheus.client.Counter;

public final class PrometheusMetricManager {

    private static final PrometheusMetricManager METRIC_MANAGER = new PrometheusMetricManager();
    private Counter inferRequestCount;
    private Counter inferLatency;
    private Counter queueLatency;

    private PrometheusMetricManager() {
        inferRequestCount =
                Counter.build()
                        .name("ts_inference_requests_total")
                        .labelNames("model_name")
                        .help("Total number of inference requests.")
                        .register();
        inferLatency =
                Counter.build()
                        .name("ts_inference_latency_microseconds")
                        .labelNames("model_name")
                        .help("Cumulative inference duration in microseconds")
                        .register();
        queueLatency =
                Counter.build()
                        .name("ts_queue_latency_microseconds")
                        .labelNames("model_name")
                        .help("Cumulative queue duration in microseconds")
                        .register();
    }

    public static PrometheusMetricManager getInstance() {
        return METRIC_MANAGER;
    }

    /**
     * Counts the time in ns it took for an inference to be completed
     *
     * @param inferTime time in nanoseconds
     * @param modelName name of the model
     */
    public void incInferLatency(long inferTime, String modelName) {
        inferLatency.labels(modelName).inc(inferTime / 1000.0);
    }

    /**
     * Counts the time in ns an inference request was queued before being executed
     *
     * @param queueTime time in nanoseconds
     * @param modelName name of the model
     */
    public void incQueueLatency(long queueTime, String modelName) {
        queueLatency.labels(modelName).inc(queueTime / 1000.0);
    }

    /**
     * Counts a valid inference request to be processed
     *
     * @param modelName name of the model
     */
    public void incInferValidCount(String modelName) {
        inferRequestCount.labels(modelName).inc();
    }
}
