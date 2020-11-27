package org.pytorch.serve.workflow.api.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.*;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.http.*;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.HttpURLConnection;
import java.util.Map;

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
        if ("wfpredict".equalsIgnoreCase(segments[1])) {
            handlePredictions(ctx, req, segments);
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private void handlePredictions(
            ChannelHandlerContext ctx, FullHttpRequest req, String[] segments)
            {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }

        predict(ctx, req, segments[2]);
    }

    private void predict(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            String wfName)
            {
        RequestInput input = parseRequest(ctx, req);
        if (wfName == null) {
                throw new BadRequestException("Parameter model_name is required.");
        }

//        Job job = new RestJob(ctx, wfName, WorkerCommands.PREDICT, input);
//        if (!ModelManager.getInstance().addJob(job)) {
//            throw new ServiceUnavailableException("Model \""+ wfName + " has no worker to serve inference request. Please use scale workers API to add workers.");
//        }
        StatusResponse status = new StatusResponse();
        status.setHttpResponseCode(HttpURLConnection.HTTP_OK);
        status.setStatus("Got inference request");
        status.setE(new Exception("All is well"));
        sendResponse(ctx, status);
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

    private static RequestInput parseRequest(
            ChannelHandlerContext ctx, FullHttpRequest req) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        RequestInput inputData = new RequestInput(requestId);

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
