package org.pytorch.serve.archive.workflow;

public class Manifest {

    private String createdOn;
    private String description;
    private String archiverVersion;
    private Workflow workflow;

    public Manifest() {
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

    public Workflow getWorkflow() {
        return workflow;
    }

    public void setWorkflow(Workflow workflow) {
        this.workflow = workflow;
    }

    public static final class Workflow {

        private String workflowName;
        private String specFile;
        private String handler;

        public Workflow() {}

        public String getWorkflowName() {
            return workflowName;
        }

        public void setWorkflowName(String workflowName) {
            this.workflowName = workflowName;
        }

        public String getSpecFile() {
            return specFile;
        }

        public void setSpecFile(String specFile) {
            this.specFile = specFile;
        }

        public String getHandler() {
            return handler;
        }

        public void setHandler(String handler) {
            this.handler = handler;
        }
    }
}
