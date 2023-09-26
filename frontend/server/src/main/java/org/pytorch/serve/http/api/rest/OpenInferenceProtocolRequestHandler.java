package org.pytorch.serve.http.api.rest;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.openapi.OpenApiUtils;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.pytorch.serve.wlm.WorkerState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.gson.JsonObject;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import com.google.gson.JsonArray;
import org.pytorch.serve.wlm.WorkerThread;

/**
 * A class handling inbound HTTP requests to the Kserve's Open Inference
 * Protocol API.
 *
 * <p>
 * This class
 */
public class OpenInferenceProtocolRequestHandler extends HttpRequestHandlerChain {

    private static final Logger logger = LoggerFactory.getLogger(OpenInferenceProtocolRequestHandler.class);
    private static final String TS_VERSION_FILE_PATH = "ts/version.txt";
    private static final String SERVER_METADATA_API = "/v2";
    private static final String SERVER_LIVE_API = "/v2/health/live";
    private static final String SERVER_READY_API = "/v2/health/ready";

    /** Creates a new {@code OpenInferenceProtocolRequestHandler} instance. */
    public OpenInferenceProtocolRequestHandler() {
    }

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
            response.addProperty("ready", ApiUtils.getTsWorkerStatus());
            NettyUtils.sendJsonResponse(ctx, response);
        } else if (concatenatedSegments.equals(SERVER_LIVE_API)) {
            // for serve live check
            JsonObject response = new JsonObject();
            response.addProperty("live", ApiUtils.getTsWorkerStatus());
            NettyUtils.sendJsonResponse(ctx, response);
        } else if (concatenatedSegments.equals(SERVER_METADATA_API)) {
            // For fetch server metadata
            JsonArray supportedExtensions = new JsonArray();
            JsonObject response = new JsonObject();
            response.addProperty("name", "Torchserve");
            response.addProperty("version", getTsVersion());
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

    private String getTsVersion() {
        String tsVersion = "";
        try {
            BufferedReader reader = new BufferedReader(new FileReader(TS_VERSION_FILE_PATH));
            String version = reader.readLine();
            reader.close();
            return version;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tsVersion;

    }
}
