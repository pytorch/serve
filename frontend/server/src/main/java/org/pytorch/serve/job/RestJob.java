package org.pytorch.serve.job;

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

public class RestJob extends Job {

    private static final Logger logger = LoggerFactory.getLogger(Job.class);
    private static final org.apache.log4j.Logger loggerTsMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    private ChannelHandlerContext ctx;

    public RestJob(
            ChannelHandlerContext ctx,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        super(modelName, version, cmd, input);
        this.ctx = ctx;
    }

    @Override
    public void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {
        long backendResponseTime = System.nanoTime() - getBegin();
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
            logMetric(InbuiltMetricsRegistry.QUEUETIME, getScheduled() - getBegin());
            logMetric(InbuiltMetricsRegistry.BACKENDRESPONSETIME, backendResponseTime);
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }
        logger.debug(
                "Waiting time ns: {}, Backend time ns: {}",
                getScheduled() - getBegin(),
                System.nanoTime() - getScheduled());
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
        }

        logger.debug(
                "Waiting time ns: {}, Inference time ns: {}",
                getScheduled() - getBegin(),
                System.nanoTime() - getBegin());
    }

    private void logMetric(String metricName, long metricValue) {
        String queueTime =
                String.valueOf(TimeUnit.MILLISECONDS.convert(metricValue, TimeUnit.NANOSECONDS));

        Dimension[] dimensions = {
            new Dimension(DimensionRegistry.LEVEL, DimensionRegistry.LevelRegistry.MODEL),
            new Dimension(DimensionRegistry.MODELNAME, getModelName()),
            new Dimension(DimensionRegistry.MODELVERSION, getModelVersion())
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
