package org.pytorch.serve.chkpnt;

import java.util.Map;

public class Checkpoint {
    private String name;
    private long created;
    private Map<String, Map<String, ModelInfo>> models;

    public Checkpoint() {}

    public Checkpoint(String chkpntName) {
        this.name = chkpntName;
        this.created = System.currentTimeMillis();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Map<String, Map<String, ModelInfo>> getModels() {
        return models;
    }

    public void setModels(Map<String, Map<String, ModelInfo>> models) {
        this.models = models;
    }

    public long getCreated() {
        return created;
    }

    public void setCreated(long created) {
        this.created = created;
    }
}

class ModelInfo {
    private boolean defaultVersion;
    private String marName;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;

    public String getMarName() {
        return marName;
    }

    public void setMarName(String marName) {
        this.marName = marName;
    }

    public boolean getDefaultVersion() {
        return defaultVersion;
    }

    public void setDefaultVersion(boolean defaultVersion) {
        this.defaultVersion = defaultVersion;
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
}
