package org.pytorch.serve.job;

import static org.pytorch.serve.util.messages.RequestInput.TS_STREAM_NEXT;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.DefaultHttpContent;
import io.netty.handler.codec.http.DefaultHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.util.CharsetUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RestJob extends Job {

    private static final Logger logger = LoggerFactory.getLogger(Job.class);

    private final IMetric inferenceLatencyMetric;
    private final IMetric queueLatencyMetric;
    private final List<String> latencyMetricDimensionValues;
    private final IMetric queueTimeMetric;
    private final List<String> queueTimeMetricDimensionValues;
    private ChannelHandlerContext ctx;
    private CompletableFuture<byte[]> responsePromise;
    /**
     * numStreams is used to track 4 cases -1: stream end 0: non-stream response (default use case)
     * 1: the first stream response [2, max_integer]: the 2nd and more stream response
     */
    private int numStreams;

    public RestJob(
            ChannelHandlerContext ctx,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        super(modelName, version, cmd, input);
        this.ctx = ctx;
        this.inferenceLatencyMetric =
                MetricCache.getInstance().getMetricFrontend("ts_inference_latency_microseconds");
        this.queueLatencyMetric =
                MetricCache.getInstance().getMetricFrontend("ts_queue_latency_microseconds");
        this.latencyMetricDimensionValues =
                Arrays.asList(
                        getModelName(),
                        getModelVersion() == null ? "default" : getModelVersion(),
                        ConfigManager.getInstance().getHostName());
        this.queueTimeMetric = MetricCache.getInstance().getMetricFrontend("QueueTime");
        this.queueTimeMetricDimensionValues =
                Arrays.asList("Host", ConfigManager.getInstance().getHostName());
        this.numStreams = 0;
    }

    @Override
    public void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {
        if (this.getCmd() == WorkerCommands.PREDICT) {
            responseInference(body, contentType, statusCode, statusPhrase, responseHeaders);
        } else if (this.getCmd() == WorkerCommands.DESCRIBE) {
            responseDescribe(body, contentType, statusCode, statusPhrase, responseHeaders);
        }
    }

    private void responseDescribe(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {
        try {
            ArrayList<DescribeModelResponse> respList =
                    ApiUtils.getModelDescription(this.getModelName(), this.getModelVersion());

            if ((body != null && body.length != 0) && respList != null && respList.size() == 1) {
                respList.get(0).setCustomizedMetadata(body);
            }

            HttpResponseStatus status =
                    (statusPhrase == null)
                            ? HttpResponseStatus.valueOf(statusCode)
                            : new HttpResponseStatus(statusCode, statusPhrase);
            FullHttpResponse resp =
                    new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);

            if (contentType != null && contentType.length() > 0) {
                resp.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
            } else {
                resp.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
            }

            if (responseHeaders != null) {
                for (Map.Entry<String, String> e : responseHeaders.entrySet()) {
                    resp.headers().set(e.getKey(), e.getValue());
                }
            }

            ByteBuf content = resp.content();
            content.writeCharSequence(JsonUtils.GSON_PRETTY.toJson(respList), CharsetUtil.UTF_8);
            content.writeByte('\n');
            NettyUtils.sendHttpResponse(ctx, resp, true);
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            logger.trace("", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, e);
        }
    }

    private void responseInference(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {
        long inferTime = System.nanoTime() - getBegin();
        HttpResponseStatus status =
                (statusPhrase == null)
                        ? HttpResponseStatus.valueOf(statusCode)
                        : new HttpResponseStatus(statusCode, statusPhrase);
        HttpResponse resp;

        if (responseHeaders != null && responseHeaders.containsKey(TS_STREAM_NEXT)) {
            resp = new DefaultHttpResponse(HttpVersion.HTTP_1_1, status, false);
            numStreams = responseHeaders.get(TS_STREAM_NEXT).equals("true") ? numStreams + 1 : -1;
        } else {
            resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
        }

        if (contentType != null && contentType.length() > 0) {
            resp.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
        }
        if (responseHeaders != null) {
            for (Map.Entry<String, String> e : responseHeaders.entrySet()) {
                resp.headers().set(e.getKey(), e.getValue());
            }
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            if (numStreams == 0) { // non-stream response
                ((DefaultFullHttpResponse) resp).content().writeBytes(body);
                NettyUtils.sendHttpResponse(ctx, resp, true);
            } else if (numStreams == -1) { // the last response in a stream
                ctx.writeAndFlush(new DefaultHttpContent(Unpooled.wrappedBuffer(body)));
                ctx.writeAndFlush(LastHttpContent.EMPTY_LAST_CONTENT);
            } else if (numStreams == 1) { // the first response in a stream
                NettyUtils.sendHttpResponse(ctx, resp, true);
                ctx.writeAndFlush(new DefaultHttpContent(Unpooled.wrappedBuffer(body)));
            } else if (numStreams > 1) { // the 2nd+ response in a stream
                ctx.writeAndFlush(new DefaultHttpContent(Unpooled.wrappedBuffer(body)));
            }
        } else if (responsePromise != null) {
            responsePromise.complete(body);
        }

        if (numStreams <= 0) {
            if (this.inferenceLatencyMetric != null) {
                try {
                    this.inferenceLatencyMetric.addOrUpdate(
                            this.latencyMetricDimensionValues, inferTime / 1000.0);
                } catch (Exception e) {
                    logger.error(
                            "Failed to update frontend metric ts_inference_latency_microseconds: ",
                            e);
                }
            }
            if (this.queueLatencyMetric != null) {
                try {
                    this.queueLatencyMetric.addOrUpdate(
                            this.latencyMetricDimensionValues,
                            (getScheduled() - getBegin()) / 1000.0);
                } catch (Exception e) {
                    logger.error(
                            "Failed to update frontend metric ts_queue_latency_microseconds: ", e);
                }
            }

            logger.debug(
                    "Waiting time ns: {}, Backend time ns: {}",
                    getScheduled() - getBegin(),
                    System.nanoTime() - getScheduled());
            double queueTime =
                    (double)
                            TimeUnit.MILLISECONDS.convert(
                                    getScheduled() - getBegin(), TimeUnit.NANOSECONDS);
            if (this.queueTimeMetric != null) {
                try {
                    this.queueTimeMetric.addOrUpdate(
                            this.queueTimeMetricDimensionValues, queueTime);
                } catch (Exception e) {
                    logger.error("Failed to update frontend metric QueueTime: ", e);
                }
            }
        }
    }

    @Override
    public void sendError(int status, String error) {
        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            // Mapping HTTPURLConnection's HTTP_ENTITY_TOO_LARGE to Netty's INSUFFICIENT_STORAGE
            status = (status == 413) ? 507 : status;
            NettyUtils.sendError(
                    ctx, HttpResponseStatus.valueOf(status), new InternalServerException(error));
        } else if (responsePromise != null) {
            responsePromise.completeExceptionally(new InternalServerException(error));
        }

        if (this.getCmd() == WorkerCommands.PREDICT) {
            logger.debug(
                    "Waiting time ns: {}, Inference time ns: {}",
                    getScheduled() - getBegin(),
                    System.nanoTime() - getBegin());
        }
    }

    public CompletableFuture<byte[]> getResponsePromise() {
        return responsePromise;
    }

    public void setResponsePromise(CompletableFuture<byte[]> responsePromise) {
        this.responsePromise = responsePromise;
    }
}
