package org.pytorch.serve.http.api.rest;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufOutputStream;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.exporter.common.TextFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PrometheusMetricsRequestHandler extends HttpRequestHandlerChain {

    private static final Logger logger =
            LoggerFactory.getLogger(PrometheusMetricsRequestHandler.class);

    /** Creates a new {@code MetricsRequestHandler} instance. */
    public PrometheusMetricsRequestHandler() {
        // TODO: Add plugins manager support
    }

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {
        if (segments.length >= 2 && "metrics".equals(segments[1])) {
            ByteBuf resBuf = Unpooled.directBuffer();
            List<String> params =
                    decoder.parameters().getOrDefault("name[]", Collections.emptyList());
            FullHttpResponse resp;
            try (OutputStream outputStream = new ByteBufOutputStream(resBuf);
                    Writer writer = new OutputStreamWriter(outputStream)) {
                TextFormat.write004(
                        writer,
                        CollectorRegistry.defaultRegistry.filteredMetricFamilySamples(
                                new HashSet<>(params)));
                resp =
                        new DefaultFullHttpResponse(
                                HttpVersion.HTTP_1_1, HttpResponseStatus.OK, resBuf);
            } catch (IOException e) {
                logger.error("Exception encountered while reporting metrics");
                throw new ModelException(e.getMessage(), e);
            }
            resp.headers().set(HttpHeaderNames.CONTENT_TYPE, TextFormat.CONTENT_TYPE_004);
            NettyUtils.sendHttpResponse(ctx, resp, true);
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }
}
