package org.pytorch.serve.snapshot;

import com.google.gson.JsonObject;
import java.util.Map;

public class Snapshot {
    private String name;
    private int modelCount;
    private long created;
    private Map<String, Map<String, JsonObject>> models;

    public Snapshot(String snaspshotName, int modelCount) {
        this.name = snaspshotName;
        this.setModelCount(modelCount);
        this.created = System.currentTimeMillis();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Map<String, Map<String, JsonObject>> getModels() {
        return models;
    }

    public void setModels(Map<String, Map<String, JsonObject>> models) {
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
