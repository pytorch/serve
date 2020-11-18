package org.pytorch.serve.workflow;

import java.util.ArrayList;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.workflow.api.http.DescribeWorkflowResponse;
import org.pytorch.serve.workflow.api.http.ListWorkflowResponse;
import org.pytorch.serve.workflow.api.http.RegisterWorkflowRequest;

public class WorkflowManager {
    private static WorkflowManager workflowManager;

    public static synchronized WorkflowManager getInstance() {
        if (workflowManager == null) {
            workflowManager = new WorkflowManager();
        }
        return workflowManager;
    }

    public StatusResponse registerModel(RegisterWorkflowRequest registerModelRequest) {
        return null;
    }

    public ListWorkflowResponse getWorkflowList(int limit, int pageToken) {
        return null;
    }

    public ArrayList<DescribeWorkflowResponse> getModelDescription(String wfName) {
        return null;
    }

    public void unregisterWorkflow(String wfName) {}
}
