package org.pytorch.serve.ensemble;

import com.google.gson.annotations.SerializedName;

public class WorkflowManifest {

    private String createdOn;
    private String description;
    private String archiverVersion;
    private RuntimeType runtime;
    private Workflow workflow;

    public WorkflowManifest() {
        runtime = RuntimeType.PYTHON;
        workflow = new Workflow();
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

    public Workflow getWorfklow() {
        return workflow;
    }

    public void setModel(Workflow model) {
        this.workflow = workflow;
    }

    public static final class Workflow {

        private String workflowName;
        private String version;
        private String specFile;
        private String workflowVersion;
        private String handler;
        private String requirementsFile;

        public Workflow() {}

        public String getWorkflowName() {
            return workflowName;
        }

        public void setWorkflowName(String workflowName) {
            this.workflowName = workflowName;
        }

        public String getVersion() {
            return version;
        }

        public void setVersion(String version) {
            this.version = version;
        }

        public String getModelVersion() {
            return workflowVersion;
        }

        public void setModelVersion(String modelVersion) {
            this.workflowVersion = modelVersion;
        }

        public String getRequirementsFile() {
            return requirementsFile;
        }

        public void setRequirementsFile(String requirementsFile) {
            this.requirementsFile = requirementsFile;
        }

        public String getHandler() {
            return handler;
        }

        public void setHandler(String handler) {
            this.handler = handler;
        }

        public String getSpecFile() {
            return specFile;
        }

        public void setSpecFile(String specFile) {
            this.specFile = specFile;
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
