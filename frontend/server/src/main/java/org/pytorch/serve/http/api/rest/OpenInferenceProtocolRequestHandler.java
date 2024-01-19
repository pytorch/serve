package org.pytorch.serve.http.api.rest;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests to the Kserve's Open Inference Protocol API.
 *
 * <p>This class
 */
public class OpenInferenceProtocolRequestHandler extends HttpRequestHandlerChain {

    private static final Logger logger =
            LoggerFactory.getLogger(OpenInferenceProtocolRequestHandler.class);
    private static final String TS_VERSION_FILE_PATH = "ts/version.txt";
    private static final String SERVER_METADATA_API = "/v2";
    private static final String SERVER_LIVE_API = "/v2/health/live";
    private static final String SERVER_READY_API = "/v2/health/ready";

    /** Creates a new {@code OpenInferenceProtocolRequestHandler} instance. */
    public OpenInferenceProtocolRequestHandler() {}

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {

        String concatenatedSegments = String.join("/", segments).trim();
        logger.info("Handling OIP http requests");
        if (concatenatedSegments.equals(SERVER_READY_API)) {
            // for serve ready check
            JsonObject response = new JsonObject();
            response.addProperty("ready", true);
            NettyUtils.sendJsonResponse(ctx, response);
        } else if (concatenatedSegments.equals(SERVER_LIVE_API)) {
            // for serve live check
            JsonObject response = new JsonObject();
            response.addProperty("live", true);
            NettyUtils.sendJsonResponse(ctx, response);
        } else if (concatenatedSegments.equals(SERVER_METADATA_API)) {
            // For fetch server metadata
            JsonArray supportedExtensions = new JsonArray();
            JsonObject response = new JsonObject();
            String tsVersion = ConfigManager.getInstance().getVersion();
            response.addProperty("name", "Torchserve");
            response.addProperty("version", tsVersion);
            supportedExtensions.add("kserve");
            supportedExtensions.add("kubeflow");
            response.add("extenstion", supportedExtensions);
            NettyUtils.sendJsonResponse(ctx, response);
        } else if (segments.length > 5 && concatenatedSegments.contains("/versions")) {
            // As of now kserve not implemented versioning, we just throws not implemented.
            JsonObject response = new JsonObject();
            response.addProperty("error", "Model versioning is not yet supported.");
            NettyUtils.sendJsonResponse(ctx, response, HttpResponseStatus.NOT_IMPLEMENTED);
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }
}
