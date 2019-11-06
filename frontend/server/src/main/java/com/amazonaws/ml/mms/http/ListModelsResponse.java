/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.http;

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
