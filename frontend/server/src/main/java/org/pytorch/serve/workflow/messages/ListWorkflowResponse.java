package org.pytorch.serve.workflow.messages;

import java.util.ArrayList;
import java.util.List;

public class ListWorkflowResponse {

    private String nextPageToken;
    private List<WorkFlowItem> workflows;

    public ListWorkflowResponse() {
        workflows = new ArrayList<>();
    }

    public String getNextPageToken() {
        return nextPageToken;
    }

    public void setNextPageToken(String nextPageToken) {
        this.nextPageToken = nextPageToken;
    }

    public List<WorkFlowItem> getWorkflows() {
        return workflows;
    }

    public void setWorkflows(List<WorkFlowItem> workflows) {
        this.workflows = workflows;
    }

    public void addModel(String workflowName, String workflowUrl) {
        workflows.add(new WorkFlowItem(workflowName, workflowUrl));
    }

    public static class WorkFlowItem {
        private String workflowName;
        private String workflowUrl;

        public WorkFlowItem(String workflowName, String workflowUrl) {
            this.workflowName = workflowName;
            this.workflowUrl = workflowUrl;
        }

        public String getWorkflowName() {
            return workflowName;
        }

        public void setWorkflowName(String workflowName) {
            this.workflowName = workflowName;
        }

        public String getWorkflowUrl() {
            return workflowUrl;
        }

        public void setWorkflowUrl(String workflowUrl) {
            this.workflowUrl = workflowUrl;
        }
    }
}
