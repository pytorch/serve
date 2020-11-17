package org.pytorch.serve.workflow.api.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.MethodNotAllowedException;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.http.messages.ListModelsResponse;
import org.pytorch.serve.http.messages.RegisterModelRequest;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerThread;

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class WorkflowMgmtRequestHandler extends HttpRequestHandlerChain {

    /** Creates a new {@code WorkflowMgmtRequestHandler} instance. */
    public WorkflowMgmtRequestHandler(Map<String, ModelServerEndpoint> ep) {
        endpointMap = ep;
    }

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        if (isManagementReq(segments)) {
            if (endpointMap.getOrDefault(segments[1], null) != null) {
                handleCustomEndpoint(ctx, req, segments, decoder);
            } else {
                if (!"workflows".equals(segments[1])) {
                    throw new ResourceNotFoundException();
                }

                HttpMethod method = req.method();
                if (segments.length < 3) {
                    if (HttpMethod.GET.equals(method)) {
                        handleListWorkflows(ctx, decoder);
                        return;
                    } else if (HttpMethod.POST.equals(method)) {
                        handleRegisterWorkflow(ctx, decoder, req);
                        return;
                    }
                    throw new MethodNotAllowedException();
                }

                String modelVersion = null;
                if (segments.length == 4) {
                    modelVersion = segments[3];
                }
                if (HttpMethod.GET.equals(method)) {
                    handleDescribeWorkflow(ctx, segments[2]);
                } else if (HttpMethod.DELETE.equals(method)) {
                    handleUnregisterWorkflow(ctx, segments[2]);
                } else {
                    throw new MethodNotAllowedException();
                }
            }
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private boolean isManagementReq(String[] segments) {
        return segments.length == 0
                || ((segments.length >= 2 && segments.length <= 4) && segments[1].equals("workflows"))
                || endpointMap.containsKey(segments[1]);
    }

    private void handleListWorkflows(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        int pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);
        if (limit > 100 || limit < 0) {
            limit = 100;
        }
        if (pageToken < 0) {
            pageToken = 0;
        }

        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Model> models = modelManager.getDefaultModels();

        List<String> keys = new ArrayList<>(models.keySet());
        Collections.sort(keys);
        ListModelsResponse list = new ListModelsResponse();

        int last = pageToken + limit;
        if (last > keys.size()) {
            last = keys.size();
        } else {
            list.setNextPageToken(String.valueOf(last));
        }

        for (int i = pageToken; i < last; ++i) {
            String modelName = keys.get(i);
            Model model = models.get(modelName);
            list.addModel(modelName, model.getModelUrl());
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleDescribeWorkflow(
            ChannelHandlerContext ctx, String modelName)
            {
//        ModelManager modelManager = ModelManager.getInstance();
//        ArrayList<DescribeModelResponse> resp = new ArrayList<DescribeModelResponse>();
//
//        if ("all".equals(modelVersion)) {
//            for (Map.Entry<String, Model> m : modelManager.getAllModelVersions(modelName)) {
//                resp.add(createModelResponse(modelManager, modelName, m.getValue()));
//            }
//        } else {
//            Model model = modelManager.getModel(modelName, modelVersion);
//            if (model == null) {
//                throw new ModelNotFoundException("Model not found: " + modelName);
//            }
//            resp.add(createModelResponse(modelManager, modelName, model));
//        }
//
//        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private DescribeModelResponse createModelResponse(
            ModelManager modelManager, String modelName, Model model) {
        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(model.getModelUrl());
        resp.setBatchSize(model.getBatchSize());
        resp.setMaxBatchDelay(model.getMaxBatchDelay());
        resp.setMaxWorkers(model.getMaxWorkers());
        resp.setMinWorkers(model.getMinWorkers());
        resp.setLoadedAtStartup(modelManager.getStartupModels().contains(modelName));
        Manifest manifest = model.getModelArchive().getManifest();
        resp.setModelVersion(manifest.getModel().getModelVersion());
        resp.setRuntime(manifest.getRuntime().getValue());

        List<WorkerThread> workers = modelManager.getWorkers(model.getModelVersionName());
        for (WorkerThread worker : workers) {
            String workerId = worker.getWorkerId();
            long startTime = worker.getStartTime();
            boolean isRunning = worker.isRunning();
            int gpuId = worker.getGpuId();
            long memory = worker.getMemory();
            int pid = worker.getPid();
            String gpuUsage = worker.getGpuUsage();
            resp.addWorker(workerId, startTime, isRunning, gpuId, memory, pid, gpuUsage);
        }

        return resp;
    }

    private void handleRegisterWorkflow(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req)
            throws ModelException {
        RegisterModelRequest registerModelRequest = parseRequest(req, decoder);
        String modelUrl = registerModelRequest.getModelUrl();
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        String modelName = registerModelRequest.getModelName();
        String runtime = registerModelRequest.getRuntime();
        String handler = registerModelRequest.getHandler();
        int batchSize = registerModelRequest.getBatchSize();
        int maxBatchDelay = registerModelRequest.getMaxBatchDelay();
        int initialWorkers = registerModelRequest.getInitialWorkers();
        boolean synchronous = registerModelRequest.getSynchronous();
        int responseTimeout = registerModelRequest.getResponseTimeout();
        if (responseTimeout == -1) {
            responseTimeout = ConfigManager.getInstance().getDefaultResponseTimeout();
        }
        Manifest.RuntimeType runtimeType = null;
        if (runtime != null) {
            try {
                runtimeType = Manifest.RuntimeType.fromValue(runtime);
            } catch (IllegalArgumentException e) {
                throw new BadRequestException(e);
            }
        }

        ModelManager modelManager = ModelManager.getInstance();
        final ModelArchive archive;
        try {

            archive =
                    modelManager.registerModel(
                            modelUrl,
                            modelName,
                            runtimeType,
                            handler,
                            batchSize,
                            maxBatchDelay,
                            responseTimeout,
                            null);
        } catch (FileAlreadyExistsException e) {
            throw new InternalServerException(
                    "Model file already exists " + FilenameUtils.getName(modelUrl), e);
        } catch (IOException | InterruptedException e) {
            throw new InternalServerException("Failed to save model: " + modelUrl, e);
        }

        modelName = archive.getModelName();

        if (initialWorkers <= 0) {
            final String msg =
                    "Model \""
                            + modelName
                            + "\" Version: "
                            + archive.getModelVersion()
                            + " registered with 0 initial workers. Use scale workers API to add workers for the model.";
            SnapshotManager.getInstance().saveSnapshot();
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
            return;
        }
    }

    private void handleUnregisterWorkflow(
            ChannelHandlerContext ctx, String modelName)
             {
//        ModelManager modelManager = ModelManager.getInstance();
//        HttpResponseStatus httpResponseStatus =
//                modelManager.unregisterModel(modelName, modelVersion);
//        if (httpResponseStatus == HttpResponseStatus.NOT_FOUND) {
//            throw new ModelNotFoundException("Model not found: " + modelName);
//        } else if (httpResponseStatus == HttpResponseStatus.BAD_REQUEST) {
//            throw new ModelVersionNotFoundException(
//                    String.format(
//                            "Model version: %s does not exist for model: %s",
//                            modelVersion, modelName));
//        } else if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
//            throw new InternalServerException("Interrupted while cleaning resources: " + modelName);
//        } else if (httpResponseStatus == HttpResponseStatus.REQUEST_TIMEOUT) {
//            throw new RequestTimeoutException("Timed out while cleaning resources: " + modelName);
//        } else if (httpResponseStatus == HttpResponseStatus.FORBIDDEN) {
//            throw new InvalidModelVersionException(
//                    "Cannot remove default version for model " + modelName);
//        }
//        String msg = "Model \"" + modelName + "\" unregistered";
//        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private RegisterModelRequest parseRequest(FullHttpRequest req, QueryStringDecoder decoder) {
        RegisterModelRequest in;
        CharSequence mime = HttpUtil.getMimeType(req);
        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(mime)) {
            in =
                    JsonUtils.GSON.fromJson(
                            req.content().toString(CharsetUtil.UTF_8), RegisterModelRequest.class);
        } else {
            in = new RegisterModelRequest(decoder);
        }
        return in;
    }
}
