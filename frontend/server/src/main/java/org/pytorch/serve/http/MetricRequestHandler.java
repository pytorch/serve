package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.*;
import org.pytorch.serve.archive.*;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import java.util.Map;


/**
 * A class handling inbound HTTP requests to the Metrics API.
 *
 */
public class MetricRequestHandler extends HttpRequestHandlerChain {

    /** Creates a new {@code MetricRequestHandler} instance. */
    public MetricRequestHandler(Map<String, ModelServerEndpoint> ep) {
        endpointMap = ep;
    }

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        if (endpointMap.getOrDefault(segments[1], null) != null) {
            handleCustomEndpoint(ctx, req, segments, decoder);
        } else{
            HttpRequestHandlerChain invalidRequestHandler = new InvalidRequestHandler();
            invalidRequestHandler.handleRequest(ctx, req, decoder, segments);
        }
    }
}
