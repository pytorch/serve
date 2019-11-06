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
package com.amazonaws.ml.mms.archive;

import com.google.gson.annotations.SerializedName;
import java.util.Map;

public class LegacyManifest {

    @SerializedName("Engine")
    private Map<String, Object> engine;

    @SerializedName("Model-Archive-Description")
    private String description;

    @SerializedName("License")
    private String license;

    @SerializedName("Model-Archive-Version")
    private String version;

    @SerializedName("Model-Server")
    private String serverVersion;

    @SerializedName("Model")
    private ModelInfo modelInfo;

    @SerializedName("Created-By")
    private CreatedBy createdBy;

    public LegacyManifest() {}

    public Map<String, Object> getEngine() {
        return engine;
    }

    public void setEngine(Map<String, Object> engine) {
        this.engine = engine;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getLicense() {
        return license;
    }

    public void setLicense(String license) {
        this.license = license;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getServerVersion() {
        return serverVersion;
    }

    public void setServerVersion(String serverVersion) {
        this.serverVersion = serverVersion;
    }

    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    public void setModelInfo(ModelInfo modelInfo) {
        this.modelInfo = modelInfo;
    }

    public CreatedBy getCreatedBy() {
        return createdBy;
    }

    public void setCreatedBy(CreatedBy createdBy) {
        this.createdBy = createdBy;
    }

    public Manifest migrate() throws InvalidModelException {
        Manifest manifest = new Manifest();
        manifest.setDescription(description);
        manifest.setLicense(license);
        manifest.setSpecificationVersion("0.1");

        if (createdBy != null) {
            Manifest.Publisher publisher = new Manifest.Publisher();
            publisher.setAuthor(createdBy.getAuthor());
            publisher.setEmail(createdBy.getEmail());
            manifest.setPublisher(publisher);
        }

        if (engine != null) {
            Object engineVersion = engine.get("MXNet");
            if (engineVersion instanceof Number) {
                Manifest.Engine eng = new Manifest.Engine();
                eng.setEngineName("MXNet");
                eng.setEngineVersion(engineVersion.toString());
                manifest.setEngine(eng);
            }
        }

        Manifest.Model model = new Manifest.Model();
        model.setModelName(modelInfo.getModelName());
        model.setDescription(modelInfo.getDescription());
        model.setHandler(modelInfo.getService());
        model.setModelVersion("snapshot");
        model.addExtension("parametersFile", modelInfo.getParameters());
        model.addExtension("symbolFile", modelInfo.getSymbol());
        manifest.setModel(model);

        if (model.getHandler() == null) {
            throw new InvalidModelException("Missing Service entry in MANIFEST.json");
        }

        return manifest;
    }

    public static final class CreatedBy {

        @SerializedName("Author")
        private String author;

        @SerializedName("Author-Email")
        private String email;

        public CreatedBy() {}

        public String getAuthor() {
            return author;
        }

        public void setAuthor(String author) {
            this.author = author;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }
    }

    public static final class ModelInfo {

        @SerializedName("Parameters")
        private String parameters;

        @SerializedName("Symbol")
        private String symbol;

        @SerializedName("Description")
        private String description;

        @SerializedName("Model-Name")
        private String modelName;

        @SerializedName("Service")
        private String service;

        public ModelInfo() {}

        public String getParameters() {
            return parameters;
        }

        public void setParameters(String parameters) {
            this.parameters = parameters;
        }

        public String getSymbol() {
            return symbol;
        }

        public void setSymbol(String symbol) {
            this.symbol = symbol;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getService() {
            return service;
        }

        public void setService(String service) {
            this.service = service;
        }
    }
}
