package org.pytorch.serve.archive.model;

import java.util.ArrayList;

public class ModelConfig {
    private int minWorkers = 1;
    private int maxWorkers = 1;
    private int batchSize = 1;
    private int maxBatchDelay = 100;
    private int responseTimeout = 120;
    private ArrayList<Integer> gpuIds;
    private int parallelLevel = 1;

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

    public int getResponseTimeout() {
        return responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }

    public ArrayList<Integer> getGpuIds() {
        return gpuIds;
    }

    public void setGpuIds(ArrayList<Integer> gpuIds) {
        this.gpuIds = gpuIds;
    }

    public int getParallelLevel() {
        return parallelLevel;
    }

    public void setParallelLevel(int parallelLevel) {
        this.parallelLevel = parallelLevel;
    }
}
