package org.pytorch.serve.plugins.endpoint;

import io.prometheus.client.Counter;
import java.util.UUID;

public final class PrometheusMetricManager {

    private static final PrometheusMetricManager METRIC_MANAGER = new PrometheusMetricManager();
    private static final String METRICS_UUID = UUID.randomUUID().toString();
    private Counter inferRequestCount;
    private Counter backendResponseLatency;
    private Counter queueLatency;
    private Counter handlerlatency;


    private PrometheusMetricManager() {
        String[] metricsLabels = {"uuid", "model_name", "model_version"};
        inferRequestCount =
                Counter.build()
                        .name("ts_inference_requests_total")
                        .labelNames(metricsLabels)
                        .help("Total number of inference requests.")
                        .register();
        backendResponseLatency =
                Counter.build()
                        .name("ts_backend_reponse_latency_milliseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative inference duration in milliseconds")
                        .register();
        queueLatency =
                Counter.build()
                        .name("ts_queue_latency_milliseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative queue duration in milliseconds")
                        .register();
        handlerlatency =
                Counter.build()
                        .name("ts_handler_latency_milliseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative handler execution duration in milliseconds")
                        .register();
    }

    private static String getOrDefaultModelVersion(String modelVersion) {
        return modelVersion == null ? "default" : modelVersion;
    }

    public static PrometheusMetricManager getInstance() {
        return METRIC_MANAGER;
    }

    /**
     * Counts the time in ms it took for an inference to be completed
     *
     * @param inferTime time in milliseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incBackendResponseLatency(long inferTime, String modelName, String modelVersion) {
        backendResponseLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(inferTime);
    }

    /**
     * Counts the time in ms an inference request was queued before being executed
     *
     * @param queueTime time in milliseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incQueueLatency(long queueTime, String modelName, String modelVersion) {
        queueLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(queueTime);
    }

    /**
     * Counts the time in ms an inference request was queued before being executed
     *
     * @param queueTime time in milliseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incHandlerLatency(long queueTime, String modelName, String modelVersion) {
        queueLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(queueTime);
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
