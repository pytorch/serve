package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.archive.model.ModelException;

public class InvalidRequestHandler extends HttpRequestHandlerChain {
    public InvalidRequestHandler() {}

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        throw new ResourceNotFoundException();
    }
}
