package org.pytorch.serve.workflow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.*;

import org.pytorch.serve.ensemble.WorkflowManifest;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.workflow.api.http.DescribeWorkflowResponse;
import org.pytorch.serve.workflow.api.http.ListWorkflowResponse;
import org.pytorch.serve.workflow.api.http.RegisterWorkflowRequest;

public class WorkflowManager {
    private static WorkflowManager workflowManager;

    private ExecutorService executorService = Executors.newFixedThreadPool(2);
    private HashMap<String, WorkflowManifest.Workflow> workflows;

    public static synchronized WorkflowManager getInstance() {
        if (workflowManager == null) {
            workflowManager = new WorkflowManager();
        }
        return workflowManager;
    }

    public Future registerWorkflow(RegisterWorkflowRequest registerWorkflowRequest) {
        FutureTask<StatusResponse> registerTask = new FutureTask<StatusResponse>(new Callable<StatusResponse>() {
            @Override
            public StatusResponse call() throws Exception {
                StatusResponse status = new StatusResponse();
                status.setHttpResponseCode(200);
                status.setStatus(String.format("Workflow {} has been registered and scaled successfully."));

                WorkflowManifest.Workflow wf = new WorkflowManifest.Workflow();
                workflows.put(registerWorkflowRequest.getWfName(), wf);
                return status;
            }
        });

        return executorService.submit(registerTask);
    }

    public ListWorkflowResponse getWorkflowList(int limit, int pageToken) {
        return null;
    }

    public ArrayList<DescribeWorkflowResponse> getWorkflowDescription(String wfName) {
        return null;
    }

    public void unregisterWorkflow(String wfName) {}
}
