package org.pytorch.serve.test.plugins.metrics;

import java.util.HashMap;

public final class TestMetricManager {

    private static final TestMetricManager METRIC_MANAGER = new TestMetricManager();


    private int inferRequestCount;
    private double backendResponseLatency;
    private double queueLatency;
    private double handlerlatency;
    private double memoryUsed;


    private TestMetricManager() {
        inferRequestCount = 0;
        backendResponseLatency = 0;
        queueLatency = 0;
        handlerlatency = 0;
        memoryUsed = 0;
    }

    public static TestMetricManager getInstance() {
        return METRIC_MANAGER;
    }


    public void incBackendResponseLatency(double value) {
        backendResponseLatency += value;
    }

    public void incQueueLatency(double value) {
        queueLatency += value;
    }


    public void incHandlerlatency(double value) {
        handlerlatency += value;
    }


    public void incMemoryUsed(double value) {
        memoryUsed += value;
    }

    public void incInferRequestCount(long value) {
        inferRequestCount += value;
    }

    public HashMap<String, String> getData() {
        HashMap<String, String> map = new HashMap<>();
        map.put("inferRequestCount", String.valueOf(inferRequestCount));
        map.put("queueLatency", String.valueOf(queueLatency));
        map.put("handlerlatency", String.valueOf(handlerlatency));
        map.put("backendResponseLatency", String.valueOf(backendResponseLatency));
        map.put("memoryUsed", String.valueOf(memoryUsed));
        return map;
    }

}
