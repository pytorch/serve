package org.pytorch.serve.archive;

import com.google.gson.annotations.SerializedName;
import java.util.LinkedHashMap;
import java.util.Map;

public class Manifest {

    private String createdOn;
    private String description;
    private String archiverVersion;
    private RuntimeType runtime;
    private Model model;

    public Manifest() {
        runtime = RuntimeType.PYTHON;
        model = new Model();
    }

    public String getCreatedOn() {
        return createdOn;
    }

    public void setCreatedOn(String createdOn) {
        this.createdOn = createdOn;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getArchiverVersion() {
        return archiverVersion;
    }

    public void setArchiverVersion(String archiverVersion) {
        this.archiverVersion = archiverVersion;
    }

    public RuntimeType getRuntime() {
        return runtime;
    }

    public void setRuntime(RuntimeType runtime) {
        this.runtime = runtime;
    }

    public Model getModel() {
        return model;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    public static final class Model {

        private String modelName;
        private String version;
        private String description;
        private String modelVersion;
        private Map<String, Object> extensions;
        private String handler;

        public Model() {}

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getVersion() {
            return version;
        }

        public void setVersion(String version) {
            this.version = version;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public String getModelVersion() {
            return modelVersion;
        }

        public void setModelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
        }

        public Map<String, Object> getExtensions() {
            return extensions;
        }

        public void setExtensions(Map<String, Object> extensions) {
            this.extensions = extensions;
        }

        public void addExtension(String key, Object value) {
            if (extensions == null) {
                extensions = new LinkedHashMap<>();
            }
            extensions.put(key, value);
        }

        public String getHandler() {
            return handler;
        }

        public void setHandler(String handler) {
            this.handler = handler;
        }
    }

    public enum RuntimeType {
        @SerializedName("python")
        PYTHON("python"),
        @SerializedName("python2")
        PYTHON2("python2"),
        @SerializedName("python3")
        PYTHON3("python3");

        String value;

        RuntimeType(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        public static RuntimeType fromValue(String value) {
            for (RuntimeType runtime : values()) {
                if (runtime.value.equals(value)) {
                    return runtime;
                }
            }
            throw new IllegalArgumentException("Invalid RuntimeType value: " + value);
        }
    }
}
