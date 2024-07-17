package org.pytorch.serve.http.api.rest;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.MethodNotAllowedException;
import org.pytorch.serve.http.RequestTimeoutException;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.ServiceUnavailableException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.http.messages.KFV1ModelReadyResponse;
import org.pytorch.serve.http.messages.ListModelsResponse;
import org.pytorch.serve.http.messages.RegisterModelRequest;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.openapi.OpenApiUtils;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.pytorch.serve.wlm.WorkerThread;

/**
 * A class handling inbound HTTP requests to the workflow management API.
 *
 * <p>This class
 */
public class ManagementRequestHandler extends HttpRequestHandlerChain {

    private ConfigManager configManager;

    /** Creates a new {@code ManagementRequestHandler} instance. */
    public ManagementRequestHandler(Map<String, ModelServerEndpoint> ep) {
        endpointMap = ep;
        configManager = ConfigManager.getInstance();
    }

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {
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
                    } else if (HttpMethod.POST.equals(method)
                            && configManager.isModelApiEnabled()) {
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
                    handleDescribeModel(ctx, req, segments[2], modelVersion, decoder);
                } else if (HttpMethod.PUT.equals(method)) {
                    if (segments.length == 5 && "set-default".equals(segments[4])) {
                        setDefaultModelVersion(ctx, segments[2], segments[3]);
                    } else {
                        handleScaleModel(ctx, decoder, segments[2], modelVersion);
                    }
                } else if (HttpMethod.DELETE.equals(method) && configManager.isModelApiEnabled()) {
                    handleUnregisterModel(ctx, segments[2], modelVersion);
                } else if (HttpMethod.OPTIONS.equals(method)) {
                    ModelManager modelManager = ModelManager.getInstance();
                    Model model = modelManager.getModel(segments[2], modelVersion);
                    if (model == null) {
                        throw new ModelNotFoundException("Model not found: " + segments[2]);
                    }

                    String resp = OpenApiUtils.getModelManagementApi(model);
                    NettyUtils.sendJsonResponse(ctx, resp);
                } else {
                    throw new MethodNotAllowedException();
                }
            }
        } else if (isKFV1ManagementReq(segments)) {
            String modelVersion = null;
            String modelName = segments[3].split(":")[0];
            HttpMethod method = req.method();
            if (HttpMethod.GET.equals(method)) {
                handleKF1ModelReady(ctx, modelName, modelVersion);
            } else {
                throw new MethodNotAllowedException();
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

    private boolean isKFV1ManagementReq(String[] segments) {
        return segments.length == 4 && "v1".equals(segments[1]) && "models".equals(segments[2]);
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        int pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);

        ListModelsResponse list = ApiUtils.getModelList(limit, pageToken);

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleDescribeModel(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            String modelName,
            String modelVersion,
            QueryStringDecoder decoder)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        boolean customizedMetadata =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "customized", "false"));
        if ("all".equals(modelVersion) || !customizedMetadata) {
            ArrayList<DescribeModelResponse> resp =
                    ApiUtils.getModelDescription(modelName, modelVersion);
            NettyUtils.sendJsonResponse(ctx, resp);
        } else {
            String requestId = NettyUtils.getRequestId(ctx.channel());
            RequestInput input = new RequestInput(requestId);
            for (Map.Entry<String, String> entry : req.headers().entries()) {
                input.updateHeaders(entry.getKey(), entry.getValue());
            }
            input.updateHeaders("describe", "True");
            RestJob job = new RestJob(ctx, modelName, modelVersion, WorkerCommands.DESCRIBE, input);
            if (!ModelManager.getInstance().addJob(job)) {
                String responseMessage = ApiUtils.getDescribeErrorResponseMessage(modelName);
                throw new ServiceUnavailableException(responseMessage);
            }
        }
    }

    private void handleKF1ModelReady(
            ChannelHandlerContext ctx, String modelName, String modelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        Model model = modelManager.getModel(modelName, modelVersion);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        KFV1ModelReadyResponse resp = createKFV1ModelReadyResponse(modelManager, modelName, model);
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private KFV1ModelReadyResponse createKFV1ModelReadyResponse(
            ModelManager modelManager, String modelName, Model model) {
        KFV1ModelReadyResponse resp = new KFV1ModelReadyResponse();
        List<WorkerThread> workers = modelManager.getWorkers(model.getModelVersionName());
        resp.setName(modelName);
        resp.setReady(!workers.isEmpty());
        return resp;
    }

    private void handleRegisterModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req)
            throws ModelException, DownloadArchiveException, WorkerInitializationException {
        RegisterModelRequest registerModelRequest = parseRequest(req, decoder);
        StatusResponse statusResponse;
        try {
            statusResponse = ApiUtils.registerModel(registerModelRequest);
        } catch (ExecutionException | InterruptedException | InternalServerException e) {
            String message;
            if (e instanceof InternalServerException) {
                message = e.getMessage();
            } else {
                message = "Error while creating workers";
            }
            statusResponse = new StatusResponse();
            statusResponse.setE(e);
            statusResponse.setStatus(message);
            statusResponse.setHttpResponseCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        }
        sendResponse(ctx, statusResponse);
    }

    private void handleUnregisterModel(
            ChannelHandlerContext ctx, String modelName, String modelVersion)
            throws ModelNotFoundException, InternalServerException, RequestTimeoutException,
                    ModelVersionNotFoundException {
        ApiUtils.unregisterModel(modelName, modelVersion);
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg, HttpResponseStatus.OK.code()));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx,
            QueryStringDecoder decoder,
            String modelName,
            String modelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException,
                    WorkerInitializationException {
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", minWorkers);
        if (modelVersion == null) {
            modelVersion = NettyUtils.getParameter(decoder, "model_version", null);
        }

        boolean synchronous =
                Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", null));

        StatusResponse statusResponse;
        try {
            statusResponse =
                    ApiUtils.updateModelWorkers(
                            modelName,
                            modelVersion,
                            minWorkers,
                            maxWorkers,
                            synchronous,
                            false,
                            null);
        } catch (ExecutionException | InterruptedException e) {
            statusResponse = new StatusResponse();
            statusResponse.setE(e);
            statusResponse.setStatus("Error while creating workers");
            statusResponse.setHttpResponseCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        }
        sendResponse(ctx, statusResponse);
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
            ChannelHandlerContext ctx, String modelName, String newModelVersion) {
        try {
            String msg = ApiUtils.setDefault(modelName, newModelVersion);
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg, HttpResponseStatus.OK.code()));
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, e);
        }
    }

    private void sendResponse(ChannelHandlerContext ctx, StatusResponse statusResponse) {
        if (statusResponse != null) {
            if (statusResponse.getHttpResponseCode() >= 200
                    && statusResponse.getHttpResponseCode() < 300) {
                NettyUtils.sendJsonResponse(ctx, statusResponse);
            } else {
                // Re-map HTTPURLConnections HTTP_ENTITY_TOO_LARGE to Netty's INSUFFICIENT_STORAGE
                int httpResponseStatus = statusResponse.getHttpResponseCode();
                NettyUtils.sendError(
                        ctx,
                        HttpResponseStatus.valueOf(
                                httpResponseStatus == 413 ? 507 : httpResponseStatus),
                        statusResponse.getE());
            }
        }
    }
}
