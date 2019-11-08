/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
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
    HttpRequestHandlerChain handlerChain;
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
        } catch (ResourceNotFoundException | ModelNotFoundException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, e);
        } catch (BadRequestException | ModelException e) {
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
