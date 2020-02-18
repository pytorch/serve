package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.checkpoint.Checkpoint;
import org.pytorch.serve.checkpoint.CheckpointManager;
import org.pytorch.serve.checkpoint.CheckpointReadException;
import org.pytorch.serve.http.messages.RegisterModelRequest;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerThread;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class ManagementRequestHandler extends HttpRequestHandlerChain {

    /** Creates a new {@code ManagementRequestHandler} instance. */
    public ManagementRequestHandler(Map<String, ModelServerEndpoint> ep) {
        endpointMap = ep;
    }

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, CheckpointReadException {
        if (CheckpointManager.getInstance().isRestartInProgress()) {
            String msg = "Restart in progress. Please try again later.";
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
            return;
        }
        if (isManagementReq(segments)) {
            if (endpointMap.getOrDefault(segments[1], null) != null) {
                handleCustomEndpoint(ctx, req, segments, decoder);
            } else {
                if (!("models".equals(segments[1]) || "checkpoints".equals(segments[1]))) {
                    throw new ResourceNotFoundException();
                }

                switch (req.method().name()) {
                    case "GET":
                        if ("models".equals(segments[1])) {
                            if (segments.length < 3) {
                                handleListModels(ctx, decoder);
                            } else {

                                String modelVersion = null;
                                if (segments.length == 4) {
                                    modelVersion = segments[3];
                                }
                                handleDescribeModel(ctx, segments[2], modelVersion);
                            }

                        } else {
                            if (segments.length == 3) {
                                getCheckpoint(ctx, segments[2]);
                            } else {
                                getAllCheckpoints(ctx);
                            }
                        }
                        break;
                    case "PUT":
                        if ("models".equals(segments[1])) {
                            if (segments.length < 3) {
                                throw new MethodNotAllowedException();
                            }
                            String modelVersion = null;
                            if (segments.length == 4) {
                                modelVersion = segments[3];
                            }
                            if (segments.length == 5 && "set-default".equals(segments[4])) {
                                setDefaultModelVersion(ctx, segments[2], segments[3]);
                            } else {
                                handleScaleModel(ctx, decoder, segments[2], modelVersion);
                            }
                        } else if (segments.length == 4 && "restart".equals(segments[3])) {
                            restartWithCheckpoint(ctx, segments[2]);
                        }
                        break;
                    case "POST":
                        if ("models".equals(segments[1])) {
                            if (segments.length < 3) {
                                handleRegisterModel(ctx, decoder, req);
                            }
                        } else {
                            if (segments.length == 3) {
                                saveCheckpoint(ctx, segments[2]);
                            }
                        }
                        break;
                    case "DELETE":
                        if ("models".equals(segments[1])) {
                            String modelVersion = null;
                            if (segments.length == 4) {
                                modelVersion = segments[3];
                            }
                            handleUnregisterModel(ctx, segments[2], modelVersion);

                        } else {
                            removeCheckpoint(ctx, segments[2]);
                        }
                        break;
                    default:
                        throw new MethodNotAllowedException();
                }
            }
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private boolean isManagementReq(String[] segments) {
        return segments.length == 0
                || ((segments.length >= 2 && segments.length <= 4) && segments[1].equals("models"))
                || (segments.length == 5 && "set-default".equals(segments[4]))
                || (segments[1].equals("checkpoints"))
                || endpointMap.containsKey(segments[1]);
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
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

    private void handleDescribeModel(
            ChannelHandlerContext ctx, String modelName, String modelVersion)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        ArrayList<DescribeModelResponse> resp = new ArrayList<DescribeModelResponse>();

        if ("all".equals(modelVersion)) {
            for (Map.Entry<Double, Model> m : modelManager.getAllModelVersions(modelName)) {
                resp.add(createModelResponse(modelManager, modelName, m.getValue()));
            }
        } else {
            Model model = modelManager.getModel(modelName, modelVersion);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            resp.add(createModelResponse(modelManager, modelName, model));
        }

        NettyUtils.sendJsonResponse(ctx, resp);
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
        Manifest.Engine engine = manifest.getEngine();
        if (engine != null) {
            resp.setEngine(engine.getEngineName());
        }
        resp.setModelVersion(manifest.getModel().getModelVersion());
        resp.setRuntime(manifest.getRuntime().getValue());

        List<WorkerThread> workers = modelManager.getWorkers(model.getModelVersionName());
        for (WorkerThread worker : workers) {
            String workerId = worker.getWorkerId();
            long startTime = worker.getStartTime();
            boolean isRunning = worker.isRunning();
            int gpuId = worker.getGpuId();
            long memory = worker.getMemory();
            resp.addWorker(workerId, startTime, isRunning, gpuId, memory);
        }

        return resp;
    }

    private void handleRegisterModel(
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
        } catch (IOException e) {
            throw new InternalServerException("Failed to save model: " + modelUrl, e);
        }

        modelName = archive.getModelName();

        final String msg = "Model \"" + modelName + "\" registered";
        if (initialWorkers <= 0) {
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
            return;
        }

        updateModelWorkers(
                ctx,
                modelName,
                archive.getModelVersion(),
                initialWorkers,
                initialWorkers,
                synchronous,
                f -> {
                    modelManager.unregisterModel(archive.getModelName(), archive.getModelVersion());
                    return null;
                });
    }

    private void handleUnregisterModel(
            ChannelHandlerContext ctx, String modelName, String modelVersion)
            throws ModelNotFoundException, InternalServerException, RequestTimeoutException,
                    ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        HttpResponseStatus httpResponseStatus =
                modelManager.unregisterModel(modelName, modelVersion);
        if (httpResponseStatus == HttpResponseStatus.NOT_FOUND) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.BAD_REQUEST) {
            throw new ModelVersionNotFoundException(
                    String.format(
                            "Model version: %s not found for model: %s", modelVersion, modelName));
        } else if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
            throw new InternalServerException("Interrupted while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.REQUEST_TIMEOUT) {
            throw new RequestTimeoutException("Timed out while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.FORBIDDEN) {
            throw new InternalServerException(
                    "Cannot remove default version for model " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.CONFLICT) {
            throw new InternalServerException(
                    "Model unregistration already in progress for model : " + modelName);
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx,
            QueryStringDecoder decoder,
            String modelName,
            String modelVersion)
            throws ModelNotFoundException {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", minWorkers);
        if (maxWorkers < minWorkers) {
            throw new BadRequestException("max_worker cannot be less than min_worker.");
        }
        boolean synchronous =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", null));

        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.getDefaultModels().containsKey(modelName)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        updateModelWorkers(ctx, modelName, modelVersion, minWorkers, maxWorkers, synchronous, null);
    }

    private void updateModelWorkers(
            final ChannelHandlerContext ctx,
            final String modelName,
            final String modelVersion,
            int minWorkers,
            int maxWorkers,
            boolean synchronous,
            final Function<Void, Void> onError) {

        ModelManager modelManager = ModelManager.getInstance();
        CompletableFuture<HttpResponseStatus> future =
                modelManager.updateModel(modelName, modelVersion, minWorkers, maxWorkers);
        if (!synchronous) {
            NettyUtils.sendJsonResponse(
                    ctx,
                    new StatusResponse("Processing worker updates..."),
                    HttpResponseStatus.ACCEPTED);
            return;
        }
        future.thenApply(
                        v -> {
                            boolean status =
                                    modelManager.scaleRequestStatus(modelName, modelVersion);
                            if (HttpResponseStatus.OK.equals(v)) {
                                if (status) {
                                    NettyUtils.sendJsonResponse(
                                            ctx, new StatusResponse("Workers scaled"), v);
                                } else {
                                    NettyUtils.sendJsonResponse(
                                            ctx,
                                            new StatusResponse("Workers scaling in progress..."),
                                            new HttpResponseStatus(210, "Partial Success"));
                                }
                            } else {
                                NettyUtils.sendError(
                                        ctx,
                                        v,
                                        new InternalServerException("Failed to start workers"));
                                if (onError != null) {
                                    onError.apply(null);
                                }
                            }
                            return v;
                        })
                .exceptionally(
                        (e) -> {
                            if (onError != null) {
                                onError.apply(null);
                            }
                            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, e);
                            return null;
                        });
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

    private void setDefaultModelVersion(
            ChannelHandlerContext ctx, String modelName, String newModelVersion)
            throws ModelNotFoundException, InternalServerException, RequestTimeoutException {
        ModelManager modelManager = ModelManager.getInstance();
        HttpResponseStatus httpResponseStatus =
                modelManager.setDefaultVersion(modelName, newModelVersion);
        if (httpResponseStatus == HttpResponseStatus.NOT_FOUND) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.FORBIDDEN) {
            throw new InternalServerException(
                    "Cannot set version " + newModelVersion + " as default for model " + modelName);
        }
        String msg =
                "Default vesion succsesfully updated for model \""
                        + modelName
                        + "\" to \""
                        + newModelVersion
                        + "\"";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void saveCheckpoint(ChannelHandlerContext ctx, String chkpntName)
            throws InternalServerException {
        CheckpointManager chkpntManager = CheckpointManager.getInstance();
        HttpResponseStatus httpResponseStatus = chkpntManager.saveCheckpoint(chkpntName);
        if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
            throw new InternalServerException("Error while saving checkpoint: " + chkpntName);
        }
        String msg = "Checkpoint " + chkpntName + " saved succsesfully";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void restartWithCheckpoint(ChannelHandlerContext ctx, String chkpntName) {
        CheckpointManager chkpntManager = CheckpointManager.getInstance();
        HttpResponseStatus httpResponseStatus = chkpntManager.restart(chkpntName);
        if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
            throw new InternalServerException("Error while starting checkpoint: " + chkpntName);
        }
        String msg = "Checkpoint " + chkpntName + " started succsesfully";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void getCheckpoint(ChannelHandlerContext ctx, String chkpntName)
            throws CheckpointReadException {
        CheckpointManager chkpntManager = CheckpointManager.getInstance();
        Checkpoint checkpoint;
        checkpoint = chkpntManager.getCheckpoint(chkpntName);
        NettyUtils.sendJsonResponse(ctx, checkpoint);
    }

    private void getAllCheckpoints(ChannelHandlerContext ctx) throws CheckpointReadException {
        CheckpointManager chkpntManager = CheckpointManager.getInstance();
        ArrayList<Checkpoint> resp;
        resp = (ArrayList<Checkpoint>) chkpntManager.getCheckpoints();
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void removeCheckpoint(ChannelHandlerContext ctx, String chkpntName)
            throws InternalServerException {
        CheckpointManager chkpntManager = CheckpointManager.getInstance();
        HttpResponseStatus httpResponseStatus = chkpntManager.removeCheckpoint(chkpntName);
        if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
            throw new InternalServerException("Error while removing checkpoint: " + chkpntName);
        }
        String msg = "Checkpoint " + chkpntName + " removed succsesfully";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }
}
