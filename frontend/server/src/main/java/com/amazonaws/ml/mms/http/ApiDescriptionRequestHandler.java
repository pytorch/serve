package com.amazonaws.ml.mms.http;

import com.amazonaws.ml.mms.archive.ModelException;
import com.amazonaws.ml.mms.openapi.OpenApiUtils;
import com.amazonaws.ml.mms.util.ConnectorType;
import com.amazonaws.ml.mms.util.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;

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
        return segments.length == 0 || segments[1].equals("api-description");
    }

    private void handleApiDescription(ChannelHandlerContext ctx) {
        NettyUtils.sendJsonResponse(ctx, OpenApiUtils.listApis(connectorType));
    }
}
