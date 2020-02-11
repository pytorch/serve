package org.pytorch.serve.checkpoint;

import java.util.Map;

public class Checkpoint {
    private String name;
    private int modelCount;
    private long created;
    private Map<String, Map<String, ModelInfo>> models;

    public Checkpoint(String chkpntName, int modelCount) {
        this.name = chkpntName;
        this.setModelCount(modelCount);
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

    @Override
    public String toString() {
        return "Checkpoint [name=" + name + ", created=" + created + ", models=" + models + "]";
    }

    public int getModelCount() {
        return modelCount;
    }

    public void setModelCount(int modelCount) {
        this.modelCount = modelCount;
    }
}

class ModelInfo {
    private boolean defaultVersion;
    private String marName;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int responseTimeout;

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

    @Override
    public String toString() {
        return "ModelInfo [defaultVersion="
                + defaultVersion
                + ", marName="
                + marName
                + ", minWorkers="
                + minWorkers
                + ", maxWorkers="
                + maxWorkers
                + ", batchSize="
                + batchSize
                + ", maxBatchDelay="
                + maxBatchDelay
                + "]";
    }

    public int getResponseTimeout() {
        return responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }
}
