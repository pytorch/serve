package org.pytorch.serve.workflow;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.*;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.ensemble.WorkFlow;
import org.pytorch.serve.ensemble.WorkflowModel;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.workflow.messages.DescribeWorkflowResponse;
import org.pytorch.serve.workflow.messages.ListWorkflowResponse;

public class WorkflowManager {
    private static WorkflowManager workflowManager;
    private final ConfigManager configManager;
    private final ConcurrentHashMap<String, WorkFlow> workflowMap;

    private WorkflowManager(ConfigManager configManager) {
        this.configManager = configManager;
        this.workflowMap = new ConcurrentHashMap<>();
    }

    public static void init(ConfigManager configManager) {
        workflowManager = new WorkflowManager(configManager);
    }

    public static synchronized WorkflowManager getInstance() {
        return workflowManager;
    }

    private WorkflowArchive createWorkflowArchive(String workflowName, String url)
            throws DownloadArchiveException, IOException, WorkflowException {
        WorkflowArchive archive =
                WorkflowArchive.downloadWorkflow(
                        configManager.getAllowedUrls(), configManager.getModelStore(), url);
        if (!(workflowName == null || workflowName.isEmpty())) {
            archive.getManifest().getWorkflow().setWorkflowName(workflowName);
        }
        archive.validate();
        return archive;
    }

    private WorkFlow createWorkflow(WorkflowArchive archive) throws Exception {
        return new WorkFlow(archive);
    }

    public StatusResponse registerWorkflow(
            String workflowName, String url, int responseTimeout, boolean synchronous)
            throws Exception {
        WorkflowArchive archive = createWorkflowArchive(workflowName, url);
        WorkFlow workflow = createWorkflow(archive);

        Map<String, Node> nodes = workflow.getDag().getNodes();

        Vector<StatusResponse> responses = new Vector<StatusResponse>();
        for (Map.Entry<String, Node> entry : nodes.entrySet()) {
            Node node = entry.getValue();
            WorkflowModel wfm = node.getWorkflowModel();

            responses.add(
                    ApiUtils.handleRegister(
                            wfm.getUrl(),
                            wfm.getName(),
                            null,
                            wfm.getHandler(),
                            wfm.getBatchSize(),
                            wfm.getMaxBatchDelay(),
                            responseTimeout,
                            wfm.getMaxWorkers(),
                            synchronous));
        }

        StatusResponse status = new StatusResponse();
        status.setHttpResponseCode(200);
        status.setStatus(
                String.format(
                        "Workflow %s has been registered and scaled successfully.", workflowName));

        workflowMap.putIfAbsent(workflowName, workflow);
        return status;
    }

    public ListWorkflowResponse getWorkflowList(int limit, int pageToken) {
        return null;
    }

    public ArrayList<DescribeWorkflowResponse> getWorkflowDescription(String wfName) {
        return null;
    }

    public void unregisterWorkflow(String wfName) {}

    public WorkFlow getWorkflow(String workflowName) {
        return workflowMap.get(workflowName);
    }
}
