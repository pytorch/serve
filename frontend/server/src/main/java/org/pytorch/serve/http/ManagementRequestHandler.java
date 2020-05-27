package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.http.messages.RegisterModelRequest;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerThread;

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
            throws ModelException {
        if (isManagementReq(segments)) {
            if (endpointMap.getOrDefault(segments[1], null) != null) {
                handleCustomEndpoint(ctx, req, segments, decoder);
            } else {
                if (!"models".equals(segments[1])) {
                    throw new ResourceNotFoundException();
                }

                HttpMethod method = req.method();
                if (segments.length < 3) {
                    if (HttpMethod.GET.equals(method)) {
                        handleListModels(ctx, decoder);
                        return;
                    } else if (HttpMethod.POST.equals(method)) {
                        handleRegisterModel(ctx, decoder, req);
                        return;
                    }
                    throw new MethodNotAllowedException();
                }

                String modelVersion = null;
                if (segments.length == 4) {
                    modelVersion = segments[3];
                }
                if (HttpMethod.GET.equals(method)) {
                    handleDescribeModel(ctx, segments[2], modelVersion);
                } else if (HttpMethod.PUT.equals(method)) {
                    if (segments.length == 5 && "set-default".equals(segments[4])) {
                        setDefaultModelVersion(ctx, segments[2], segments[3]);
                    } else {
                        handleScaleModel(ctx, decoder, segments[2], modelVersion);
                    }
                } else if (HttpMethod.DELETE.equals(method)) {
                    handleUnregisterModel(ctx, segments[2], modelVersion);
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
                || ((segments.length >= 2 && segments.length <= 4) && segments[1].equals("models"))
                || (segments.length == 5 && "set-default".equals(segments[4]))
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
            throws ModelNotFoundException, ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        ArrayList<DescribeModelResponse> resp = new ArrayList<DescribeModelResponse>();

        if ("all".equals(modelVersion)) {
            for (Map.Entry<String, Model> m : modelManager.getAllModelVersions(modelName)) {
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
        } catch (FileAlreadyExistsException e) {
            throw new InternalServerException(
                    "Model file already exists " + FilenameUtils.getName(modelUrl), e);
        } catch (IOException e) {
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

        updateModelWorkers(
                ctx,
                modelName,
                archive.getModelVersion(),
                initialWorkers,
                initialWorkers,
                synchronous,
                true,
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
                            "Model version: %s does not exist for model: %s",
                            modelVersion, modelName));
        } else if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR) {
            throw new InternalServerException("Interrupted while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.REQUEST_TIMEOUT) {
            throw new RequestTimeoutException("Timed out while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.FORBIDDEN) {
            throw new InvalidModelVersionException(
                    "Cannot remove default version for model " + modelName);
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx,
            QueryStringDecoder decoder,
            String modelName,
            String modelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", minWorkers);
        if (modelVersion == null) {
            modelVersion = NettyUtils.getParameter(decoder, "model_version", null);
        }
        if (maxWorkers < minWorkers) {
            throw new BadRequestException("max_worker cannot be less than min_worker.");
        }
        boolean synchronous =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", null));

        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.getDefaultModels().containsKey(modelName)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        updateModelWorkers(
                ctx, modelName, modelVersion, minWorkers, maxWorkers, synchronous, false, null);
    }

    private void updateModelWorkers(
            final ChannelHandlerContext ctx,
            final String modelName,
            final String modelVersion,
            int minWorkers,
            int maxWorkers,
            boolean synchronous,
            boolean isInit,
            final Function<Void, Void> onError)
            throws ModelVersionNotFoundException {
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
                                    String msg =
                                            "Workers scaled to "
                                                    + minWorkers
                                                    + " for model: "
                                                    + modelName;
                                    if (modelVersion != null) {
                                        msg += ", version: " + modelVersion; // NOPMD
                                    }

                                    if (isInit) {
                                        msg =
                                                "Model \""
                                                        + modelName
                                                        + "\" Version: "
                                                        + modelVersion
                                                        + " registered with "
                                                        + minWorkers
                                                        + " initial workers";
                                    }

                                    NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg), v);
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
            throws ModelNotFoundException, InternalServerException, RequestTimeoutException,
                    ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        HttpResponseStatus httpResponseStatus =
                modelManager.setDefaultVersion(modelName, newModelVersion);
        if (httpResponseStatus == HttpResponseStatus.NOT_FOUND) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.FORBIDDEN) {
            throw new ModelVersionNotFoundException(
                    "Model version " + newModelVersion + " does not exist for model " + modelName);
        }
        String msg =
                "Default vesion succsesfully updated for model \""
                        + modelName
                        + "\" to \""
                        + newModelVersion
                        + "\"";
        SnapshotManager.getInstance().saveSnapshot();
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }
}
