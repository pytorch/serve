package org.pytorch.serve.ensemble;

public class WorkflowModel {

    private String name;
    private String url;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int timeOutMs;
    private int retryAttempts;
    private String handler;

    public WorkflowModel(
            String name,
            String url,
            int minWorkers,
            int maxWorkers,
            int batchSize,
            int maxBatchDelay,
            int retryAttempts,
            int timeOutMs,
            String handler) {
        this.name = name;
        this.url = url;
        this.minWorkers = minWorkers;
        this.maxWorkers = maxWorkers;
        this.batchSize = batchSize;
        this.maxBatchDelay = maxBatchDelay;
        this.retryAttempts = retryAttempts;
        this.timeOutMs = timeOutMs;
        this.handler = handler;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public int getMinWorkers() {
        return minWorkers;
    }

    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    public int getRetryAttempts() {
        return retryAttempts;
    }

    public void setRetryAttempts(int retryAttempts) {
        this.retryAttempts = retryAttempts;
    }

    public int getTimeOutMs() {
        return timeOutMs;
    }

    public void setTimeOutMs(int timeOutMs) {
        this.timeOutMs = timeOutMs;
    }

    public String getHandler() {
        return handler;
    }
}
