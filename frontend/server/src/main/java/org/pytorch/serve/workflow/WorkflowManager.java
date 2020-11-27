package org.pytorch.serve.workflow;

import io.netty.channel.ChannelHandlerContext;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.workflow.InvalidWorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.ensemble.InvalidDAGException;
import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.ensemble.WorkFlow;
import org.pytorch.serve.ensemble.WorkflowModel;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class WorkflowManager {
    private static final Logger logger = LoggerFactory.getLogger(WorkflowManager.class);
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

    private WorkFlow createWorkflow(WorkflowArchive archive)
            throws IOException, InvalidDAGException, InvalidWorkflowException {
        return new WorkFlow(archive);
    }

    public StatusResponse registerWorkflow(
            String workflowName, String url, int responseTimeout, boolean synchronous)
            throws IOException, ExecutionException, InterruptedException {
        StatusResponse status = new StatusResponse();
        try {
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

            status.setHttpResponseCode(HttpURLConnection.HTTP_OK);
            status.setStatus(
                    String.format(
                            "Workflow %s has been registered and scaled successfully.",
                            workflowName));

            workflowMap.putIfAbsent(workflowName, workflow);
        } catch (DownloadArchiveException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Failed to download workflow archive file");
            status.setE(e);
        } catch (WorkflowException | InvalidDAGException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Invalid workflow specification");
            status.setE(e);
        } catch (ModelException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Failed to register workflow models");
            status.setE(e);
        }

        return status;
    }

    public ConcurrentHashMap<String, WorkFlow> getWorkflows() {
        return workflowMap;
    }

    public void unregisterWorkflow(String workflowName) {
        WorkFlow workflow = workflowMap.get(workflowName);
        Map<String, Node> nodes = workflow.getDag().getNodes();
        for (Map.Entry<String, Node> entry : nodes.entrySet()) {
            Node node = entry.getValue();
            WorkflowModel wfm = node.getWorkflowModel();
            new Thread(
                            () -> {
                                try {
                                    ApiUtils.unregisterModel(wfm.getName(), null);
                                } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
                                    logger.error(
                                            "Could not unregister workflow model: " + wfm.getName(),
                                            e);
                                }
                            })
                    .start();
        }

        workflowMap.remove(workflowName);
        WorkflowArchive.removeWorkflow(workflowName, workflow.getWorkflowArchive().getUrl());
    }

    public WorkFlow getWorkflow(String workflowName) {
        return workflowMap.get(workflowName);
    }

    public void predict(ChannelHandlerContext ctx, String wfName, RequestInput input) {}
}
