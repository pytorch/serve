package org.pytorch.serve.ensemble;

public class Node {

    private String name;
    private String parentName;
    private WorkflowModel workflowModel;

    public Node(String name, WorkflowModel model) {
        this.name = name;
        this.workflowModel = model;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getParentName() {
        return parentName;
    }

    public void setParentName(String parentName) {
        this.parentName = parentName;
    }

    public WorkflowModel getWorkflowModel() {
        return workflowModel;
    }
}
