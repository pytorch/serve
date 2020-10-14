package org.pytorch.serve.plugins.endpoint.prometheus;

import io.prometheus.client.Counter;
import io.prometheus.client.Gauge;
import io.prometheus.client.Histogram;
import java.util.UUID;

/**
 * This class registers different metrics with Prometheus registry and provides wrapper methods to
 * update the metric values
 */
public final class PrometheusMetricManager {

    private static final PrometheusMetricManager METRIC_MANAGER = new PrometheusMetricManager();
    private static final String METRICS_UUID = UUID.randomUUID().toString();
    private Counter inferRequestCount;
    private Histogram backendResponseLatency;
    private Histogram queueLatency;
    private Histogram handlerlatency;
    private Gauge memoryUsed;

    private PrometheusMetricManager() {
        String[] metricsLabels = {"uuid", "model_name", "model_version"};
        inferRequestCount =
                Counter.build()
                        .name("ts_inference_requests_total")
                        .labelNames(metricsLabels)
                        .help("Total number of inference requests.")
                        .register();

        memoryUsed =
                Gauge.build()
                        .name("memory_used")
                        .labelNames(new String[] {"uuid"})
                        .help("System Memory used")
                        .register();

        backendResponseLatency =
                Histogram.build()
                        .name("ts_backend_reponse_latency_milliseconds")
                        .labelNames(metricsLabels)
                        .help("histogram of Backend response duration in milliseconds")
                        .register();
        queueLatency =
                Histogram.build()
                        .name("ts_queue_latency_milliseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative Queue duration in milliseconds")
                        .register();
        handlerlatency =
                Histogram.build()
                        .name("ts_handler_latency_milliseconds")
                        .labelNames(metricsLabels)
                        .help("Cumulative Handler execution duration in milliseconds")
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
    public void incBackendResponseLatency(double inferTime, String modelName, String modelVersion) {
        backendResponseLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .observe(inferTime);
    }

    /**
     * Counts the time in ms an inference request was queued before being executed
     *
     * @param queueTime time in milliseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incQueueLatency(double queueTime, String modelName, String modelVersion) {
        queueLatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .observe(queueTime);
    }

    /**
     * Counts the time in ms an inference request was queued before being executed
     *
     * @param handlerTime time in milliseconds
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incHandlerLatency(double handlerTime, String modelName, String modelVersion) {
        handlerlatency
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .observe(handlerTime);
    }

    /**
     * Counts a valid inference request to be processed
     *
     * @param value value to increment
     * @param modelName name of the model
     * @param modelVersion version of the model
     */
    public void incInferCount(long value, String modelName, String modelVersion) {
        inferRequestCount
                .labels(METRICS_UUID, modelName, getOrDefaultModelVersion(modelVersion))
                .inc(value);
    }

    /**
     * System memory used
     *
     * @param memoryUsedValue name of the model
     */
    public void addMemoryUsed(double memoryUsedValue) {
        memoryUsed.labels(METRICS_UUID).set(memoryUsedValue);
    }
}
