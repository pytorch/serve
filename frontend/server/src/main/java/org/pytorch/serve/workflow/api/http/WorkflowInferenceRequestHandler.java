package org.pytorch.serve.workflow.api.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.net.HttpURLConnection;
import java.util.List;
import java.util.Map;

import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.ServiceUnavailableException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.metrics.api.MetricAggregator;
import org.pytorch.serve.openapi.OpenApiUtils;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests to the workflow inference API.
 *
 * <p>This class
 */
public class WorkflowInferenceRequestHandler extends HttpRequestHandlerChain {

    private static final Logger logger =
            LoggerFactory.getLogger(org.pytorch.serve.http.api.rest.InferenceRequestHandler.class);

    /** Creates a new {@code WorkflowInferenceRequestHandler} instance. */
    public WorkflowInferenceRequestHandler() {}

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException {
        if (isInferenceReq(segments)) {
            if (endpointMap.getOrDefault(segments[1], null) != null) {
                handleCustomEndpoint(ctx, req, segments, decoder);
            } else {
                switch (segments[1]) {
                    case "ping":
                        Runnable r =
                                () -> {
                                    String response = ApiUtils.getWorkerStatus();
                                    NettyUtils.sendJsonResponse(
                                            ctx,
                                            new StatusResponse(
                                                    response, HttpURLConnection.HTTP_OK));
                                };
                        ApiUtils.getTorchServeHealth(r);
                        break;
                    case "predictions":
                        handlePredictions(ctx, req, segments);
                        break;
                    default:
                        throw new ResourceNotFoundException();
                }
            }
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private boolean isInferenceReq(String[] segments) {
        return segments.length == 0
                || (segments.length >= 2
                        && (segments[1].equals("ping")
                                || segments[1].equals("predictions")
                                || segments[1].equals("api-description")
                                || endpointMap.containsKey(segments[1])))
                || (segments.length == 3 && segments[2].equals("predict"))
                || (segments.length == 4 && segments[3].equals("predict"));
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }

        String modelVersion = null;

        if (segments.length == 4) {
            modelVersion = segments[3];
        }
        predict(ctx, req, null, segments[2], modelVersion);
    }

    private void predict(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String modelName,
            String modelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        RequestInput input = parseRequest(ctx, req, decoder);
        if (modelName == null) {
            modelName = input.getStringParameter("model_name");
            if (modelName == null) {
                throw new BadRequestException("Parameter model_name is required.");
            }
        }

        if (HttpMethod.OPTIONS.equals(req.method())) {
            ModelManager modelManager = ModelManager.getInstance();

            Model model = modelManager.getModel(modelName, modelVersion);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }

            String resp = OpenApiUtils.getModelApi(model);
            NettyUtils.sendJsonResponse(ctx, resp);
            return;
        }

        MetricAggregator.handleInferenceMetric(modelName, modelVersion);
        Job job = new RestJob(ctx, modelName, modelVersion, WorkerCommands.PREDICT, input);
        if (!ModelManager.getInstance().addJob(job)) {
            String responseMessage =
                    "Model \""
                            + modelName
                            + "\" Version "
                            + modelVersion
                            + " has no worker to serve inference request. Please use scale workers API to add workers.";

            if (modelVersion == null) {
                responseMessage =
                        "Model \""
                                + modelName
                                + "\" has no worker to serve inference request. Please use scale workers API to add workers.";
            }

            throw new ServiceUnavailableException(responseMessage);
        }
    }

    private static RequestInput parseRequest(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        RequestInput inputData = new RequestInput(requestId);
        if (decoder != null) {
            for (Map.Entry<String, List<String>> entry : decoder.parameters().entrySet()) {
                String key = entry.getKey();
                for (String value : entry.getValue()) {
                    inputData.addParameter(new InputParameter(key, value));
                }
            }
        }

        CharSequence contentType = HttpUtil.getMimeType(req);
        for (Map.Entry<String, String> entry : req.headers().entries()) {
            inputData.updateHeaders(entry.getKey(), entry.getValue());
        }

        if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            HttpDataFactory factory = new DefaultHttpDataFactory(6553500);
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(factory, req);
            try {
                while (form.hasNext()) {
                    inputData.addParameter(NettyUtils.getFormData(form.next()));
                }
            } catch (HttpPostRequestDecoder.EndOfDataDecoderException ignore) {
                logger.trace("End of multipart items.");
            } finally {
                form.cleanFiles();
                form.destroy();
            }
        } else {
            byte[] content = NettyUtils.getBytes(req.content());
            inputData.addParameter(new InputParameter("body", content, contentType));
        }
        return inputData;
    }
}
