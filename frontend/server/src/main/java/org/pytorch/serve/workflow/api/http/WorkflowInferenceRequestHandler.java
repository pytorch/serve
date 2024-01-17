package org.pytorch.serve.workflow.api.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.util.Map;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowNotFoundException;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.pytorch.serve.workflow.WorkflowManager;
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

    private static RequestInput parseRequest(ChannelHandlerContext ctx, FullHttpRequest req) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        RequestInput inputData = new RequestInput(requestId);

        CharSequence contentType = HttpUtil.getMimeType(req);
        for (Map.Entry<String, String> entry : req.headers().entries()) {
            inputData.updateHeaders(entry.getKey(), entry.getValue());
        }

        if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            HttpDataFactory factory =
                    new DefaultHttpDataFactory(ConfigManager.getInstance().getMaxRequestSize());
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

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {

        if ("wfpredict".equalsIgnoreCase(segments[1])) {
            if (segments.length < 3) {
                throw new ResourceNotFoundException();
            }
            handlePredictions(ctx, req, segments);
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments)
            throws WorkflowNotFoundException {
        RequestInput input = parseRequest(ctx, req);
        logger.info(input.toString());
        String wfName = segments[2];
        if (wfName == null) {
            throw new BadRequestException("Parameter workflow_name is required.");
        }
        WorkflowManager.getInstance().predict(ctx, wfName, input);
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
