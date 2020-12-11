package org.pytorch.serve.http;

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
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.metrics.api.MetricAggregator;
import org.pytorch.serve.openapi.OpenApiUtils;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
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
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class InferenceRequestHandler extends HttpRequestHandlerChain {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);

    /** Creates a new {@code InferenceRequestHandler} instance. */
    public InferenceRequestHandler(Map<String, ModelServerEndpoint> ep) {
        endpointMap = ep;
    }

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
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
                    case "models":
                    case "invocations":
                        validatePredictionsEndpoint(segments);
                        handleInvocations(ctx, req, decoder, segments);
                        break;
                    case "predictions":
                        handlePredictions(ctx, req, segments, false);
                        break;
                    case "explanations":
                        handlePredictions(ctx, req, segments, true);
                        break;
                    default:
                        handleLegacyPredict(ctx, req, decoder, segments);
                        break;
                }
            }
        } else if (isKFV1InferenceReq(segments)) {
            if (segments[3].contains(":predict")) {
                handleKFV1Predictions(ctx, req, segments, false);
            } else if (segments[3].contains(":explain")) {
                handleKFV1Predictions(ctx, req, segments, true);
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
                                || segments[1].equals("explanations")
                                || segments[1].equals("api-description")
                                || segments[1].equals("invocations")
                                || endpointMap.containsKey(segments[1])))
                || (segments.length == 4 && segments[1].equals("models"))
                || (segments.length == 3 && segments[2].equals("predict"))
                || (segments.length == 4 && segments[3].equals("predict"));
    }

    private boolean isKFV1InferenceReq(String[] segments) {
        return segments.length == 4
                && "v1".equals(segments[1])
                && "models".equals(segments[2])
                && (segments[3].contains(":predict") || segments[3].contains(":explain"));
    }

    private void validatePredictionsEndpoint(String[] segments) {
        if (segments.length == 2 && "invocations".equals(segments[1])) {
            return;
        } else if (segments.length == 4
                && "models".equals(segments[1])
                && "invoke".equals(segments[3])) {
            return;
        }

        throw new ResourceNotFoundException();
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments, boolean explain)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }

        String modelVersion = null;

        if (segments.length == 4) {
            modelVersion = segments[3];
        }
        req.headers().add("explain", "False");
        if (explain) {
            req.headers().add("explain", "True");
        }

        predict(ctx, req, null, segments[2], modelVersion);
    }

    private void handleKFV1Predictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments, boolean explain)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        String modelVersion = null;
        String modelName = segments[3].split(":")[0];

        req.headers().add("explain", "False");
        if (explain) {
            req.headers().add("explain", "True");
        }

        predict(ctx, req, null, modelName, modelVersion);
    }

    private void handleInvocations(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        String modelName =
                ("invocations".equals(segments[1]))
                        ? NettyUtils.getParameter(decoder, "model_name", null)
                        : segments[2];
        if (modelName == null || modelName.isEmpty()) {
            if (ModelManager.getInstance().getStartupModels().size() == 1) {
                modelName = ModelManager.getInstance().getStartupModels().iterator().next();
            }
        }
        predict(ctx, req, decoder, modelName, null);
    }

    private void handleLegacyPredict(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException, ModelVersionNotFoundException {

        String modelVersion = null;
        if (segments.length == 4 && "predict".equals(segments[3])) {
            modelVersion = segments[2];
        } else if (segments.length < 3 || !"predict".equals(segments[2])) {
            throw new ResourceNotFoundException();
        }

        predict(ctx, req, decoder, segments[1], modelVersion);
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
                    ApiUtils.getInferenceErrorResponseMessage(modelName, modelVersion);

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
