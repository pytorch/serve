package org.pytorch.serve.http.api.rest;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.MethodNotAllowedException;
import org.pytorch.serve.openapi.OpenApiUtils;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.WorkerInitializationException;

public class ApiDescriptionRequestHandler extends HttpRequestHandlerChain {

    private ConnectorType connectorType;

    public ApiDescriptionRequestHandler(ConnectorType type) {
        connectorType = type;
    }

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {
        if (isApiDescription(segments)) {
            String path = decoder.path();
            if (("/".equals(path) && HttpMethod.OPTIONS.equals(req.method()))
                    || (segments.length == 2 && segments[1].equals("api-description"))) {
                handleApiDescription(ctx);
                return;
            }
            throw new MethodNotAllowedException();
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private boolean isApiDescription(String[] segments) {
        return segments.length == 0
                || (segments.length == 2 && segments[1].equals("api-description"));
    }

    private void handleApiDescription(ChannelHandlerContext ctx) {
        NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listApis(connectorType));
    }
}
