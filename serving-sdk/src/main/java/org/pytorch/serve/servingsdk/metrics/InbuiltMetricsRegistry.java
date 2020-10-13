package org.pytorch.serve.servingsdk.metrics;

/**
 * This is a registry for different Metrics available in Torch Serve.
 * This is not exhaustive list.
 * A Metric can be created or logged without adding it's entry to this registry.
 */
public class InbuiltMetricsRegistry {
    public static final String INFERENCE = "Inference";
    public static final String QUEUETIME = "QueueTime";
    public static final String BACKENDRESPONSETIME = "BackendResponseTime";
    public static final String HANDLERTIME = "HandlerTime";
    public static final String WORKERTHREADTIME = "WorkerThreadTime";
    public static final String WORKERLOADTIME = "WorkerLoadTime";
    public static final String REQUESTS2XX = "Requests2XX";
    public static final String REQUESTS4XX = "Requests4XX";
    public static final String REQUESTS5XX = "Requests5XX";
}
