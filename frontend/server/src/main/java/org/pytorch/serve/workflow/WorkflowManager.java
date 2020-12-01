package org.pytorch.serve.workflow;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.workflow.InvalidWorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.ensemble.DagExecutor;
import org.pytorch.serve.ensemble.InvalidDAGException;
import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.ensemble.NodeOutput;
import org.pytorch.serve.ensemble.WorkFlow;
import org.pytorch.serve.ensemble.WorkflowModel;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.workflow.messages.ModelRegistrationResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class WorkflowManager {
    private static final Logger logger = LoggerFactory.getLogger(WorkflowManager.class);

    private final ExecutorService inferenceExecutorService =
            Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

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
                        configManager.getAllowedUrls(), configManager.getWorkflowStore(), url);
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
            String workflowName, String url, int responseTimeout, boolean synchronous) {
        StatusResponse status = new StatusResponse();
        ExecutorService executorService = Executors.newFixedThreadPool(4);
        CompletionService<ModelRegistrationResult> executorCompletionService =
                new ExecutorCompletionService<>(executorService);
        boolean failed = false;
        ArrayList<String> failedMessages = new ArrayList<>();
        ArrayList<String> successNodes = new ArrayList<>();
        try {
            WorkflowArchive archive = createWorkflowArchive(workflowName, url);
            WorkFlow workflow = createWorkflow(archive);

            Map<String, Node> nodes = workflow.getDag().getNodes();

            List<Future<ModelRegistrationResult>> futures = new ArrayList<>();

            for (Map.Entry<String, Node> entry : nodes.entrySet()) {
                Node node = entry.getValue();
                WorkflowModel wfm = node.getWorkflowModel();

                futures.add(
                        executorCompletionService.submit(
                                () -> registerModelWrapper(wfm, responseTimeout, synchronous)));
            }

            int i = 0;
            while (i < futures.size()) {
                i++;
                Future<ModelRegistrationResult> future = executorCompletionService.take();
                if (future.isCancelled()) {
                    failed = true;
                    continue;
                }

                ModelRegistrationResult result = future.get();
                if (result.getResponse().getHttpResponseCode() != HttpURLConnection.HTTP_OK) {
                    failed = true;
                    failedMessages.add(result.getResponse().getStatus());
                    for (Future<ModelRegistrationResult> f : futures) {
                        f.cancel(false);
                    }
                } else {
                    successNodes.add(result.getModelName());
                }
            }

            if (failed) {
                status.setHttpResponseCode(HttpURLConnection.HTTP_INTERNAL_ERROR);
                String message =
                        String.format(
                                "Workflow %s has failed to register. Failures: %s",
                                workflow.getWorkflowArchive().getWorkflowName(),
                                failedMessages.toString());
                status.setStatus(message);
                status.setE(new WorkflowException(message));

                Map<String, Node> unregisterNodes = new HashMap<>();
                for (String nodeName : successNodes) {
                    unregisterNodes.put(nodeName, nodes.get(nodeName));
                }
                unregisterModels(unregisterNodes);

            } else {
                status.setHttpResponseCode(HttpURLConnection.HTTP_OK);
                status.setStatus(
                        String.format(
                                "Workflow %s has been registered and scaled successfully.",
                                workflow.getWorkflowArchive().getWorkflowName()));

                workflowMap.putIfAbsent(workflow.getWorkflowArchive().getWorkflowName(), workflow);
            }

        } catch (DownloadArchiveException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Failed to download workflow archive file");
            status.setE(e);
        } catch (WorkflowException | InvalidDAGException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Invalid workflow specification");
            status.setE(e);
        } catch (Exception e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Failed to register workflow");
            status.setE(e);
        } finally {
            executorService.shutdown();
        }
        return status;
    }

    public ModelRegistrationResult registerModelWrapper(
            WorkflowModel wfm, int responseTimeout, boolean synchronous) {
        StatusResponse status = new StatusResponse();
        try {
            status =
                    ApiUtils.handleRegister(
                            wfm.getUrl(),
                            wfm.getName(),
                            null,
                            wfm.getHandler(),
                            wfm.getBatchSize(),
                            wfm.getMaxBatchDelay(),
                            responseTimeout,
                            wfm.getMaxWorkers(),
                            synchronous,
                            true);
        } catch (Exception e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_INTERNAL_ERROR);
            status.setStatus(
                    String.format(
                            "Workflow Node %s failed to register. %s",
                            wfm.getName(), e.getMessage()));
        }

        return new ModelRegistrationResult(wfm.getName(), status);
    }

    public ConcurrentHashMap<String, WorkFlow> getWorkflows() {
        return workflowMap;
    }

    public void unregisterWorkflow(String workflowName) {
        WorkFlow workflow = workflowMap.get(workflowName);
        Map<String, Node> nodes = workflow.getDag().getNodes();
        unregisterModels(nodes);
        workflowMap.remove(workflowName);
        WorkflowArchive.removeWorkflow(workflowName, workflow.getWorkflowArchive().getUrl());
    }

    public void unregisterModels(Map<String, Node> nodes) {

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
    }

    public WorkFlow getWorkflow(String workflowName) {
        return workflowMap.get(workflowName);
    }

    public void predict(ChannelHandlerContext ctx, String wfName, RequestInput input) {
        WorkFlow wf = workflowMap.get(wfName);
        if (wf != null) {
            DagExecutor dagExecutor = new DagExecutor(wf.getDag());
            CompletableFuture<NodeOutput> predictionFuture =
                    CompletableFuture.supplyAsync(() -> dagExecutor.execute(input));
            predictionFuture
                    .thenApplyAsync(
                            (prediction) -> {
                                if (prediction != null && prediction.getData() != null) {
                                    FullHttpResponse resp =
                                            new DefaultFullHttpResponse(
                                                    HttpVersion.HTTP_1_1,
                                                    HttpResponseStatus.OK,
                                                    false);
                                    resp.headers()
                                            .set(
                                                    HttpHeaderNames.CONTENT_TYPE,
                                                    HttpHeaderValues.APPLICATION_JSON);
                                    resp.content().writeBytes((byte[]) prediction.getData());
                                    NettyUtils.sendHttpResponse(ctx, resp, true);
                                } else {
                                    throw new InternalServerException(
                                            "Workflow inference request failed!");
                                }
                                return null;
                            },
                            inferenceExecutorService)
                    .exceptionally(
                            ex -> {
                                NettyUtils.sendError(
                                        ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, ex);
                                return null;
                            });
        } else {
            throw new ResourceNotFoundException();
        }
    }
}
