package org.pytorch.serve.servingsdk.metrics;

/**
 * This is a registry for different Metrics available in TorchServe.
 * The purpose of this registry is to list and refer all the Metrics available in TorchServe in one place.
 * It is expected thar whenever a new metrics get added in TorchServe, a corresponding entry should be made here.
 * Plugin writers then will be able to refer this list.
 * This is not exhaustive list. A Metric can be created or logged without adding it's entry to this registry.
 * */
public class InbuiltMetricsRegistry {
    public static final String INFERENCEREQUESTS = "InferenceRequests";
    public static final String QUEUETIME = "QueueTime";
    public static final String BACKENDRESPONSETIME = "BackendResponseTime";
    public static final String HANDLERTIME = "HandlerTime";
    public static final String WORKERTHREADTIME = "WorkerThreadTime";
    public static final String WORKERLOADTIME = "WorkerLoadTime";
    public static final String REQUESTS2XX = "Requests2XX";
    public static final String REQUESTS4XX = "Requests4XX";
    public static final String REQUESTS5XX = "Requests5XX";

    // System metrics collected from ts/metrics/system_metrics.py
    public static final String CPUUTILIZATION = "CPUUtilization";
    public static final String MEMORYUSED = "MemoryUsed";
    public static final String MEMORYAVAILABLE = "MemoryAvailable";
    public static final String MEMORYUTILIZATION = "MemoryUtilization";
    public static final String DISKUSAGE = "DiskUsage";
    public static final String DISKUTILIZATION = "DiskUtilization";
    public static final String DISKAVAILABLE = "DiskAvailable";


}
