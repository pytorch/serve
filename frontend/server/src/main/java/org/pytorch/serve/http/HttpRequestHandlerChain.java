package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.util.Map;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.ModelServerEndpointException;
import org.pytorch.serve.servingsdk.impl.ModelServerContext;
import org.pytorch.serve.servingsdk.impl.ModelServerRequest;
import org.pytorch.serve.servingsdk.impl.ModelServerResponse;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class HttpRequestHandlerChain {
    private static final Logger logger = LoggerFactory.getLogger(HttpRequestHandler.class);
    protected Map<String, ModelServerEndpoint> endpointMap;
    protected HttpRequestHandlerChain chain;

    public HttpRequestHandlerChain() {}

    public HttpRequestHandlerChain(Map<String, ModelServerEndpoint> map) {
        endpointMap = map;
    }

    public HttpRequestHandlerChain setNextHandler(HttpRequestHandlerChain nextHandler) {
        chain = nextHandler;
        return chain;
    }

    public abstract void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException, ModelException, DownloadArchiveException,
                    WorkflowException, WorkerInitializationException;

    private void run(
            ModelServerEndpoint endpoint,
            FullHttpRequest req,
            FullHttpResponse rsp,
            QueryStringDecoder decoder,
            String method)
            throws IOException {
        switch (method) {
            case "GET":
                endpoint.doGet(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            case "PUT":
                endpoint.doPut(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            case "DELETE":
                endpoint.doDelete(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            case "POST":
                endpoint.doPost(
                        new ModelServerRequest(req, decoder),
                        new ModelServerResponse(rsp),
                        new ModelServerContext());
                break;
            default:
                throw new ServiceUnavailableException("Invalid HTTP method received");
        }
    }

    protected void handleCustomEndpoint(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            String[] segments,
            QueryStringDecoder decoder) {
        ModelServerEndpoint endpoint = endpointMap.get(segments[1]);
        Runnable r =
                () -> {
                    Long start = System.currentTimeMillis();
                    FullHttpResponse rsp =
                            new DefaultFullHttpResponse(
                                    HttpVersion.HTTP_1_1, HttpResponseStatus.OK, true);
                    try {
                        run(endpoint, req, rsp, decoder, req.method().toString());
                        NettyUtils.sendHttpResponse(ctx, rsp, true);
                        logger.info(
                                "Running \"{}\" endpoint took {} ms",
                                segments[0],
                                System.currentTimeMillis() - start);
                    } catch (ModelServerEndpointException me) {
                        NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, me);
                        logger.error("Error thrown by the model endpoint plugin.", me);
                    } catch (OutOfMemoryError oom) {
                        NettyUtils.sendError(
                                ctx, HttpResponseStatus.INSUFFICIENT_STORAGE, oom, "Out of memory");
                    } catch (IOException ioe) {
                        NettyUtils.sendError(
                                ctx,
                                HttpResponseStatus.INTERNAL_SERVER_ERROR,
                                ioe,
                                "I/O error while running the custom endpoint");
                        logger.error("I/O error while running the custom endpoint.", ioe);
                    } catch (Throwable e) {
                        NettyUtils.sendError(
                                ctx,
                                HttpResponseStatus.INTERNAL_SERVER_ERROR,
                                e,
                                "Unknown exception");
                        logger.error("Unknown exception", e);
                    }
                };
        ModelManager.getInstance().submitTask(r);
    }
}
