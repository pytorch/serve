package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.workflow.WorkflowNotFoundException;
import org.pytorch.serve.util.NettyUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests.
 *
 * <p>This class
 */
public class HttpRequestHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Logger logger = LoggerFactory.getLogger(HttpRequestHandler.class);
    private HttpRequestHandlerChain handlerChain;
    /** Creates a new {@code HttpRequestHandler} instance. */
    public HttpRequestHandler() {}

    public HttpRequestHandler(HttpRequestHandlerChain chain) {
        handlerChain = chain;
    }

    /** {@inheritDoc} */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest req) {
        try {
            NettyUtils.requestReceived(ctx.channel(), req);
            if (!req.decoderResult().isSuccess()) {
                throw new BadRequestException("Invalid HTTP message.");
            }
            QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
            String path = decoder.path();

            String[] segments = path.split("/");
            handlerChain.handleRequest(ctx, req, decoder, segments);
        } catch (ResourceNotFoundException
                | ModelNotFoundException
                | ModelVersionNotFoundException
                | WorkflowNotFoundException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, e);
        } catch (BadRequestException | ModelException | DownloadArchiveException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, e);
        } catch (ConflictStatusException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.CONFLICT, e);
        } catch (RequestTimeoutException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.REQUEST_TIMEOUT, e);
        } catch (MethodNotAllowedException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.METHOD_NOT_ALLOWED, e);
        } catch (ServiceUnavailableException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.SERVICE_UNAVAILABLE, e);
        } catch (OutOfMemoryError e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.INSUFFICIENT_STORAGE, e);
        } catch (IllegalArgumentException e) {
            logger.error("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.FORBIDDEN, e);
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("", cause);
        if (cause instanceof OutOfMemoryError) {
            NettyUtils.sendError(ctx, HttpResponseStatus.INSUFFICIENT_STORAGE, cause);
        }
        ctx.close();
    }
}
