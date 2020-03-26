package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.openapi.OpenApiUtils;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.NettyUtils;

public class ApiDescriptionRequestHandler extends HttpRequestHandlerChain {

    private ConnectorType connectorType;

    public ApiDescriptionRequestHandler(ConnectorType type) {
        connectorType = type;
    }

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {

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
