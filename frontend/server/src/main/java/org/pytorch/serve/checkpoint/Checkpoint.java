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