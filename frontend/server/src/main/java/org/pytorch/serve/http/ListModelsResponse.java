package org.pytorch.serve.http;

import java.util.ArrayList;
import java.util.List;

public class ListModelsResponse {

    private String nextPageToken;
    private List<ModelItem> models;

    public ListModelsResponse() {
        models = new ArrayList<>();
    }

    public String getNextPageToken() {
        return nextPageToken;
    }

    public void setNextPageToken(String nextPageToken) {
        this.nextPageToken = nextPageToken;
    }

    public List<ModelItem> getModels() {
        return models;
    }

    public void setModels(List<ModelItem> models) {
        this.models = models;
    }

    public void addModel(String modelName, String modelUrl) {
        models.add(new ModelItem(modelName, modelUrl));
    }

    public static final class ModelItem {

        private String modelName;
        private String modelUrl;

        public ModelItem() {}

        public ModelItem(String modelName, String modelUrl) {
            this.modelName = modelName;
            this.modelUrl = modelUrl;
        }

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getModelUrl() {
            return modelUrl;
        }

        public void setModelUrl(String modelUrl) {
            this.modelUrl = modelUrl;
        }
    }
}
