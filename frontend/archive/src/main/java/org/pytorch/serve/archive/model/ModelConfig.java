package org.pytorch.serve.archive.model;

import java.util.ArrayList;

public class ModelConfig {
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int responseTimeout;
    private CoreType coreType = CoreType.NONE;
    private ArrayList<Integer> coreIds;
    private int parallelLevel = 1;
    private ParallelType parallelType = ParallelType.NONE;

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

    public ArrayList<Integer> getCoreIds() {
        return coreIds;
    }

    public void setCoreIds(ArrayList<Integer> coreIds) {
        this.coreIds = coreIds;
    }

    public int getParallelLevel() {
        return parallelLevel;
    }

    public void setParallelLevel(int parallelLevel) {
        this.parallelLevel = parallelLevel;
    }

    public void setParallelType(String parallelType) {
        this.parallelType = ParallelType.valueOf(parallelType);
    }

    public ParallelType getParallelType() {
        return parallelType;
    }

    public void setCoreType(String coreType) {
        this.coreType = CoreType.valueOf(coreType);
    }

    public CoreType getCoreType() {
        return coreType;
    }

    public enum ParallelType {
        NONE(""),
        PP("pp"),
        TP("tp"),
        PPTP("pptp");

        private String type;

        ParallelType(String type) {
            this.type = type;
        }

        public String getParallelType() {
            return type;
        }
    }

    public enum CoreType {
        NONE(""),
        CPU("cpu"),
        GPU("gpu"),
        NEURON("neuron");

        private String type;

        CoreType(String type) {
            this.type = type;
        }

        public String getCoreType() {
            return type;
        }
    }
}
