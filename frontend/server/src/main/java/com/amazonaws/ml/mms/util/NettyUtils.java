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
package com.amazonaws.ml.mms.util;

import com.amazonaws.ml.mms.http.ErrorResponse;
import com.amazonaws.ml.mms.http.Session;
import com.amazonaws.ml.mms.metrics.Dimension;
import com.amazonaws.ml.mms.metrics.Metric;
import com.amazonaws.ml.mms.util.messages.InputParameter;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.Attribute;
import io.netty.handler.codec.http.multipart.FileUpload;
import io.netty.handler.codec.http.multipart.InterfaceHttpData;
import io.netty.util.AttributeKey;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.net.SocketAddress;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A utility class that handling Netty request and response. */
public final class NettyUtils {

    private static final Logger logger = LoggerFactory.getLogger("ACCESS_LOG");

    private static final String REQUEST_ID = "x-request-id";
    private static final AttributeKey<Session> SESSION_KEY = AttributeKey.valueOf("session");
    private static final Dimension DIMENSION = new Dimension("Level", "Host");
    private static final Metric REQUESTS_2_XX =
            new Metric(
                    "Requests2XX",
                    "1",
                    "Count",
                    ConfigManager.getInstance().getHostName(),
                    DIMENSION);
    private static final Metric REQUESTS_4_XX =
            new Metric(
                    "Requests4XX",
                    "1",
                    "Count",
                    ConfigManager.getInstance().getHostName(),
                    DIMENSION);
    private static final Metric REQUESTS_5_XX =
            new Metric(
                    "Requests5XX",
                    "1",
                    "Count",
                    ConfigManager.getInstance().getHostName(),
                    DIMENSION);

    private static final org.apache.log4j.Logger loggerMmsMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    private NettyUtils() {}

    public static void requestReceived(Channel channel, HttpRequest request) {
        Session session = channel.attr(SESSION_KEY).get();
        assert session == null;

        SocketAddress address = channel.remoteAddress();
        String remoteIp;
        if (address == null) {
            // This is can be null on UDS, or on certain case in Windows
            remoteIp = "0.0.0.0";
        } else {
            remoteIp = address.toString();
        }
        channel.attr(SESSION_KEY).set(new Session(remoteIp, request));
    }

    public static String getRequestId(Channel channel) {
        Session accessLog = channel.attr(SESSION_KEY).get();
        if (accessLog != null) {
            return accessLog.getRequestId();
        }
        return null;
    }

    public static void sendJsonResponse(ChannelHandlerContext ctx, Object json) {
        sendJsonResponse(ctx, JsonUtils.GSON_PRETTY.toJson(json), HttpResponseStatus.OK);
    }

    public static void sendJsonResponse(
            ChannelHandlerContext ctx, Object json, HttpResponseStatus status) {
        sendJsonResponse(ctx, JsonUtils.GSON_PRETTY.toJson(json), status);
    }

    public static void sendJsonResponse(ChannelHandlerContext ctx, String json) {
        sendJsonResponse(ctx, json, HttpResponseStatus.OK);
    }

    public static void sendJsonResponse(
            ChannelHandlerContext ctx, String json, HttpResponseStatus status) {
        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
        resp.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        ByteBuf content = resp.content();
        content.writeCharSequence(json, CharsetUtil.UTF_8);
        content.writeByte('\n');
        sendHttpResponse(ctx, resp, true);
    }

    public static void sendError(
            ChannelHandlerContext ctx, HttpResponseStatus status, Throwable t) {
        ErrorResponse error =
                new ErrorResponse(status.code(), t.getClass().getSimpleName(), t.getMessage());
        sendJsonResponse(ctx, error, status);
    }

    public static void sendError(
            ChannelHandlerContext ctx, HttpResponseStatus status, Throwable t, String msg) {
        ErrorResponse error = new ErrorResponse(status.code(), t.getClass().getSimpleName(), msg);
        sendJsonResponse(ctx, error, status);
    }

    /**
     * Send HTTP response to client.
     *
     * @param ctx ChannelHandlerContext
     * @param resp HttpResponse to send
     * @param keepAlive if keep the connection
     */
    public static void sendHttpResponse(
            ChannelHandlerContext ctx, FullHttpResponse resp, boolean keepAlive) {
        // Send the response and close the connection if necessary.
        Channel channel = ctx.channel();
        Session session = channel.attr(SESSION_KEY).getAndSet(null);
        HttpHeaders headers = resp.headers();

        ConfigManager configManager = ConfigManager.getInstance();
        if (session != null) {
            // session might be recycled if channel is closed already.
            session.setCode(resp.status().code());
            headers.set(REQUEST_ID, session.getRequestId());
            logger.info(session.toString());
        }
        int code = resp.status().code();
        if (code >= 200 && code < 300) {
            loggerMmsMetrics.info(REQUESTS_2_XX);
        } else if (code >= 400 && code < 500) {
            loggerMmsMetrics.info(REQUESTS_4_XX);
        } else {
            loggerMmsMetrics.info(REQUESTS_5_XX);
        }

        String allowedOrigin = configManager.getCorsAllowedOrigin();
        String allowedMethods = configManager.getCorsAllowedMethods();
        String allowedHeaders = configManager.getCorsAllowedHeaders();

        if (allowedOrigin != null
                && !allowedOrigin.isEmpty()
                && !headers.contains(HttpHeaderNames.ACCESS_CONTROL_ALLOW_ORIGIN)) {
            headers.set(HttpHeaderNames.ACCESS_CONTROL_ALLOW_ORIGIN, allowedOrigin);
        }
        if (allowedMethods != null
                && !allowedMethods.isEmpty()
                && !headers.contains(HttpHeaderNames.ACCESS_CONTROL_ALLOW_METHODS)) {
            headers.set(HttpHeaderNames.ACCESS_CONTROL_ALLOW_METHODS, allowedMethods);
        }
        if (allowedHeaders != null
                && !allowedHeaders.isEmpty()
                && !headers.contains(HttpHeaderNames.ACCESS_CONTROL_ALLOW_HEADERS)) {
            headers.set(HttpHeaderNames.ACCESS_CONTROL_ALLOW_HEADERS, allowedHeaders);
        }

        // Add cache-control headers to avoid browser cache response
        headers.set("Pragma", "no-cache");
        headers.set("Cache-Control", "no-cache; no-store, must-revalidate, private");
        headers.set("Expires", "Thu, 01 Jan 1970 00:00:00 UTC");

        HttpUtil.setContentLength(resp, resp.content().readableBytes());
        if (!keepAlive || code >= 400) {
            headers.set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
            ChannelFuture f = channel.writeAndFlush(resp);
            f.addListener(ChannelFutureListener.CLOSE);
        } else {
            headers.set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
            channel.writeAndFlush(resp);
        }
    }

    /** Closes the specified channel after all queued write requests are flushed. */
    public static void closeOnFlush(Channel ch) {
        if (ch.isActive()) {
            ch.writeAndFlush(Unpooled.EMPTY_BUFFER).addListener(ChannelFutureListener.CLOSE);
        }
    }

    public static byte[] getBytes(ByteBuf buf) {
        if (buf.hasArray()) {
            return buf.array();
        }

        byte[] ret = new byte[buf.readableBytes()];
        int readerIndex = buf.readerIndex();
        buf.getBytes(readerIndex, ret);
        return ret;
    }

    public static String getParameter(QueryStringDecoder decoder, String key, String def) {
        List<String> param = decoder.parameters().get(key);
        if (param != null && !param.isEmpty()) {
            return param.get(0);
        }
        return def;
    }

    public static int getIntParameter(QueryStringDecoder decoder, String key, int def) {
        String value = getParameter(decoder, key, null);
        if (value == null) {
            return def;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return def;
        }
    }

    public static InputParameter getFormData(InterfaceHttpData data) {
        if (data == null) {
            return null;
        }

        String name = data.getName();
        switch (data.getHttpDataType()) {
            case Attribute:
                Attribute attribute = (Attribute) data;
                try {
                    return new InputParameter(name, attribute.getValue());
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            case FileUpload:
                FileUpload fileUpload = (FileUpload) data;
                String contentType = fileUpload.getContentType();
                try {
                    return new InputParameter(name, getBytes(fileUpload.getByteBuf()), contentType);
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            default:
                throw new IllegalArgumentException(
                        "Except form field, but got " + data.getHttpDataType());
        }
    }
}
