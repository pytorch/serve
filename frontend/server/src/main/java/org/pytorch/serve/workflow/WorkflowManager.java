package org.pytorch.serve.workflow;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.workflow.InvalidWorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowNotFoundException;
import org.pytorch.serve.ensemble.DagExecutor;
import org.pytorch.serve.ensemble.InvalidDAGException;
import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.ensemble.NodeOutput;
import org.pytorch.serve.ensemble.WorkFlow;
import org.pytorch.serve.ensemble.WorkflowModel;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.http.InternalServerException;
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

    private final ThreadFactory namedThreadFactory =
            new ThreadFactoryBuilder().setNameFormat("wf-manager-thread-%d").build();
    private final ExecutorService inferenceExecutorService =
            Executors.newFixedThreadPool(
                    Runtime.getRuntime().availableProcessors(), namedThreadFactory);

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
        return createWorkflowArchive(workflowName, url, false);
    }

    private WorkflowArchive createWorkflowArchive(
            String workflowName, String url, boolean s3SseKmsEnabled)
            throws DownloadArchiveException, IOException, WorkflowException {
        WorkflowArchive archive =
                WorkflowArchive.downloadWorkflow(
                        configManager.getAllowedUrls(),
                        configManager.getWorkflowStore(),
                        url,
                        s3SseKmsEnabled);
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
            String workflowName,
            String url,
            int responseTimeout,
            int startupTimeout,
            boolean synchronous)
            throws WorkflowException {
        return registerWorkflow(
                workflowName, url, responseTimeout, startupTimeout, synchronous, false);
    }

    public StatusResponse registerWorkflow(
            String workflowName,
            String url,
            int responseTimeout,
            int startupTimeout,
            boolean synchronous,
            boolean s3SseKms)
            throws WorkflowException {

        if (url == null) {
            throw new BadRequestException("Parameter url is required.");
        }

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

            if (workflowMap.get(workflow.getWorkflowArchive().getWorkflowName()) != null) {
                throw new ConflictStatusException(
                        "Workflow "
                                + workflow.getWorkflowArchive().getWorkflowName()
                                + " is already registered.");
            }

            Map<String, Node> nodes = workflow.getDag().getNodes();

            List<Future<ModelRegistrationResult>> futures = new ArrayList<>();

            for (Map.Entry<String, Node> entry : nodes.entrySet()) {
                Node node = entry.getValue();
                WorkflowModel wfm = node.getWorkflowModel();

                futures.add(
                        executorCompletionService.submit(
                                () ->
                                        registerModelWrapper(
                                                wfm,
                                                responseTimeout,
                                                startupTimeout,
                                                synchronous)));
            }

            int i = 0;
            while (i < futures.size()) {
                i++;
                Future<ModelRegistrationResult> future = executorCompletionService.take();
                ModelRegistrationResult result = future.get();
                if (result.getResponse().getHttpResponseCode() != HttpURLConnection.HTTP_OK) {
                    failed = true;
                    String msg;
                    if (result.getResponse().getStatus() == null) {
                        msg =
                                "Failed to register the model "
                                        + result.getModelName()
                                        + ". Check error logs.";
                    } else {
                        msg = result.getResponse().getStatus();
                    }
                    failedMessages.add(msg);
                } else {
                    successNodes.add(result.getModelName());
                }
            }

            if (failed) {
                String rollbackFailure = null;
                try {
                    removeArtifacts(workflowName, workflow, successNodes);
                } catch (Exception e) {
                    rollbackFailure =
                            "Error while doing rollback of failed workflow. Details"
                                    + e.getMessage();
                }

                if (rollbackFailure != null) {
                    failedMessages.add(rollbackFailure);
                }
                status.setHttpResponseCode(HttpURLConnection.HTTP_INTERNAL_ERROR);
                String message =
                        String.format(
                                "Workflow %s has failed to register. Failures: %s",
                                workflow.getWorkflowArchive().getWorkflowName(),
                                failedMessages.toString());
                status.setStatus(message);
                status.setE(new WorkflowException(message));

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
        } catch (InvalidDAGException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_BAD_REQUEST);
            status.setStatus("Invalid workflow specification");
            status.setE(e);
        } catch (InterruptedException | ExecutionException | IOException e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_INTERNAL_ERROR);
            status.setStatus("Failed to register workflow.");
            status.setE(e);
        } finally {
            executorService.shutdown();
        }
        return status;
    }

    public ModelRegistrationResult registerModelWrapper(
            WorkflowModel wfm, int responseTimeout, int startupTimeout, boolean synchronous) {
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
                            startupTimeout,
                            wfm.getMaxWorkers(),
                            synchronous,
                            true,
                            false);
        } catch (Exception e) {
            status.setHttpResponseCode(HttpURLConnection.HTTP_INTERNAL_ERROR);
            String msg;
            if (e.getMessage() == null) {
                msg = "Check error logs.";
            } else {
                msg = e.getMessage();
            }
            status.setStatus(
                    String.format(
                            "Workflow Node %s failed to register. Details: %s",
                            wfm.getName(), msg));
            status.setE(e);
            logger.error("Model '" + wfm.getName() + "' failed to register.", e);
        }

        return new ModelRegistrationResult(wfm.getName(), status);
    }

    public ConcurrentHashMap<String, WorkFlow> getWorkflows() {
        return workflowMap;
    }

    public void unregisterWorkflow(String workflowName, ArrayList<String> successNodes)
            throws WorkflowNotFoundException, InterruptedException, ExecutionException {

        WorkFlow workflow = workflowMap.get(workflowName);
        if (workflow == null) {
            throw new WorkflowNotFoundException("Workflow not found: " + workflowName);
        }
        workflowMap.remove(workflowName);
        removeArtifacts(workflowName, workflow, successNodes);
    }

    public void removeArtifacts(
            String workflowName, WorkFlow workflow, ArrayList<String> successNodes)
            throws ExecutionException, InterruptedException {
        WorkflowArchive.removeWorkflow(
                configManager.getWorkflowStore(), workflow.getWorkflowArchive().getUrl());
        Map<String, Node> nodes = workflow.getDag().getNodes();
        unregisterModels(workflowName, nodes, successNodes);
    }

    public void unregisterModels(
            String workflowName, Map<String, Node> nodes, ArrayList<String> successNodes)
            throws InterruptedException, ExecutionException {

        ExecutorService executorService = Executors.newFixedThreadPool(4);
        CompletionService<ModelRegistrationResult> executorCompletionService =
                new ExecutorCompletionService<>(executorService);
        List<Future<ModelRegistrationResult>> futures = new ArrayList<>();

        for (Map.Entry<String, Node> entry : nodes.entrySet()) {
            Node node = entry.getValue();
            WorkflowModel wfm = node.getWorkflowModel();

            futures.add(
                    executorCompletionService.submit(
                            () -> {
                                StatusResponse status = new StatusResponse();
                                try {
                                    ApiUtils.unregisterModel(wfm.getName(), null);
                                    status.setHttpResponseCode(HttpURLConnection.HTTP_OK);
                                    status.setStatus(
                                            String.format(
                                                    "Unregisterd workflow node %s", wfm.getName()));
                                } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
                                    if (successNodes == null
                                            || successNodes.contains(wfm.getName())) {
                                        status.setHttpResponseCode(
                                                HttpURLConnection.HTTP_INTERNAL_ERROR);
                                        status.setStatus(
                                                String.format(
                                                        "Error while unregistering workflow node %s",
                                                        wfm.getName()));
                                        status.setE(e);
                                        logger.error(
                                                "Model '"
                                                        + wfm.getName()
                                                        + "' failed to unregister.",
                                                e);
                                    } else {
                                        status.setHttpResponseCode(HttpURLConnection.HTTP_OK);
                                        status.setStatus(
                                                String.format(
                                                        "Error while unregistering workflow node %s but can be ignored.",
                                                        wfm.getName()));
                                        status.setE(e);
                                    }
                                } catch (Exception e) {
                                    status.setHttpResponseCode(
                                            HttpURLConnection.HTTP_INTERNAL_ERROR);
                                    status.setStatus(
                                            String.format(
                                                    "Error while unregistering workflow node %s",
                                                    wfm.getName()));
                                    status.setE(e);
                                }
                                return new ModelRegistrationResult(wfm.getName(), status);
                            }));
        }

        int i = 0;
        boolean failed = false;
        ArrayList<String> failedMessages = new ArrayList<>();
        while (i < futures.size()) {
            i++;
            Future<ModelRegistrationResult> future = executorCompletionService.take();
            ModelRegistrationResult result = future.get();
            if (result.getResponse().getHttpResponseCode() != HttpURLConnection.HTTP_OK) {
                failed = true;
                failedMessages.add(result.getResponse().getStatus());
            }
        }

        if (failed) {
            throw new InternalServerException(
                    "Error while unregistering the workflow "
                            + workflowName
                            + ". Details: "
                            + failedMessages.toArray().toString());
        }
        executorService.shutdown();
    }

    public WorkFlow getWorkflow(String workflowName) {
        return workflowMap.get(workflowName);
    }

    public void predict(ChannelHandlerContext ctx, String wfName, RequestInput input)
            throws WorkflowNotFoundException {
        WorkFlow wf = workflowMap.get(wfName);
        if (wf != null) {
            DagExecutor dagExecutor = new DagExecutor(wf.getDag());
            CompletableFuture<ArrayList<NodeOutput>> predictionFuture =
                    CompletableFuture.supplyAsync(() -> dagExecutor.execute(input, null));
            predictionFuture
                    .thenApplyAsync(
                            (predictions) -> {
                                if (!predictions.isEmpty()) {
                                    if (predictions.size() == 1) {
                                        FullHttpResponse resp =
                                                new DefaultFullHttpResponse(
                                                        HttpVersion.HTTP_1_1,
                                                        HttpResponseStatus.OK,
                                                        true);
                                        resp.headers()
                                                .set(
                                                        HttpHeaderNames.CONTENT_TYPE,
                                                        HttpHeaderValues.APPLICATION_JSON);
                                        resp.content()
                                                .writeBytes((byte[]) predictions.get(0).getData());
                                        NettyUtils.sendHttpResponse(ctx, resp, true);

                                    } else {
                                        JsonObject result = new JsonObject();
                                        for (NodeOutput prediction : predictions) {
                                            String val =
                                                    new String(
                                                            (byte[]) prediction.getData(),
                                                            StandardCharsets.UTF_8);
                                            result.add(
                                                    prediction.getNodeName(),
                                                    JsonParser.parseString(val).getAsJsonObject());
                                        }
                                        NettyUtils.sendJsonResponse(ctx, result);
                                    }
                                } else {
                                    throw new InternalServerException(
                                            "Workflow inference request failed!");
                                }
                                return null;
                            },
                            inferenceExecutorService)
                    .exceptionally(
                            ex -> {
                                String[] error = ex.getMessage().split(":");
                                NettyUtils.sendError(
                                        ctx,
                                        HttpResponseStatus.INTERNAL_SERVER_ERROR,
                                        new InternalServerException(
                                                error[error.length - 1].strip()));
                                return null;
                            });
        } else {
            throw new WorkflowNotFoundException("Workflow not found: " + wfName);
        }
    }
}
