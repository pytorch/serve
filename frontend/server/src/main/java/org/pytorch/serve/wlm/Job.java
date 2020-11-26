package org.pytorch.serve.wlm;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.servingsdk.metrics.DimensionRegistry;
import org.pytorch.serve.servingsdk.metrics.InbuiltMetricsRegistry;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Job {

    private static final Logger logger = LoggerFactory.getLogger(Job.class);
    private static final org.apache.log4j.Logger loggerTsMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    private ChannelHandlerContext ctx;

    private String modelName;
    private String modelVersion;
    private WorkerCommands cmd; // Else its data msg or inf requests
    private RequestInput input;
    private long begin;
    private long scheduled;

    public Job(
            ChannelHandlerContext ctx,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        this.ctx = ctx;
        this.modelName = modelName;
        this.cmd = cmd;
        this.input = input;
        this.modelVersion = version;
        begin = System.nanoTime();
        scheduled = begin;
    }

    public String getJobId() {
        return input.getRequestId();
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public WorkerCommands getCmd() {
        return cmd;
    }

    public boolean isControlCmd() {
        return !WorkerCommands.PREDICT.equals(cmd);
    }

    public RequestInput getPayload() {
        return input;
    }

    public void setScheduled() {
        scheduled = System.nanoTime();
    }

    public void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {
        long backendResponseTime = System.nanoTime() - scheduled;
        HttpResponseStatus status =
                (statusPhrase == null)
                        ? HttpResponseStatus.valueOf(statusCode)
                        : new HttpResponseStatus(statusCode, statusPhrase);
        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);

        if (contentType != null && contentType.length() > 0) {
            resp.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
        }
        if (responseHeaders != null) {
            for (Map.Entry<String, String> e : responseHeaders.entrySet()) {
                resp.headers().set(e.getKey(), e.getValue());
            }
        }
        resp.content().writeBytes(body);

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            logMetric(InbuiltMetricsRegistry.QUEUETIME, scheduled - begin);
            logMetric(InbuiltMetricsRegistry.BACKENDRESPONSETIME, backendResponseTime);
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }
        logger.debug(
                "Waiting time ns: {}, Backend time ns: {}",
                scheduled - begin,
                System.nanoTime() - scheduled);
    }

    public void sendError(HttpResponseStatus status, String error) {
        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendError(ctx, status, new InternalServerException(error));
        }

        logger.debug(
                "Waiting time ns: {}, Inference time ns: {}",
                scheduled - begin,
                System.nanoTime() - begin);
    }

    private void logMetric(String metricName, long metricValue) {
        String queueTime =
                String.valueOf(TimeUnit.MILLISECONDS.convert(metricValue, TimeUnit.NANOSECONDS));

        Dimension[] dimensions = {
            new Dimension(DimensionRegistry.LEVEL, DimensionRegistry.LevelRegistry.MODEL),
            new Dimension(DimensionRegistry.MODELNAME, modelName),
            new Dimension(DimensionRegistry.MODELVERSION, modelVersion)
        };

        loggerTsMetrics.info(
                new Metric(
                        metricName,
                        queueTime,
                        "ms",
                        ConfigManager.getInstance().getHostName(),
                        dimensions));
    }
}
