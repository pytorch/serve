package org.pytorch.serve.workflow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.*;

import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.ensemble.Dag;
import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.ensemble.WorkflowManifest;
import org.pytorch.serve.ensemble.WorkflowModel;
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

    public StatusResponse registerWorkflow(RegisterWorkflowRequest registerWorkflowRequest) {
        StatusResponse status = new StatusResponse();
        status.setHttpResponseCode(200);
        status.setStatus(String.format("Workflow {} has been registered and scaled successfully."));

        WorkflowManifest.Workflow wf = new WorkflowManifest.Workflow();
        workflows.put(registerWorkflowRequest.getWfName(), wf);
        //register(dag, registerWorkflowRequest.getResponseTimeout(), true);

        return status;
    }


    public ListWorkflowResponse getWorkflowList(int limit, int pageToken) {
        return null;
    }

    public ArrayList<DescribeWorkflowResponse> getWorkflowDescription(String wfName) {
        return null;
    }

    public void unregisterWorkflow(String wfName) {}
}
