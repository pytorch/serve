package org.pytorch.serve.plugins.endpoint;

import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import org.pytorch.serve.servingsdk.*;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;


import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.exporter.common.TextFormat;

import org.pytorch.serve.servingsdk.SingletonAppender;
import org.pytorch.serve.servingsdk.LogEvent;
import org.pytorch.serve.servingsdk.LogEventListener;


import java.io.IOException;

import java.io.Writer;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;


@Endpoint(
        urlPattern = "metrics",
        endpointType = EndpointTypes.METRIC,
        description = "Prometheus Metric endpoint")
public class PrometheusMetrics extends ModelServerEndpoint {

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        Properties prop = ctx.getConfig();
        Map<String, List<String>> params_map = req.getParameterMap();
        List<String> params = params_map.getOrDefault("name[]", Collections.emptyList());


        Writer writer = new StringWriter();
        TextFormat.write004(writer,CollectorRegistry.defaultRegistry.filteredMetricFamilySamples(
                new HashSet<>(params)));


        // 6 * 1024 * 1024
        int maxRequestSize = Integer.parseInt(prop.getProperty("max_request_size", "6291456"));

        rsp.setContentType(TextFormat.CONTENT_TYPE_004);
        rsp.getOutputStream()
                .write(writer.toString().getBytes(StandardCharsets.UTF_8));
    }

    public void register(SingletonAppender singletonAppender) {
        LogEventListenerImpl logEventListener = new LogEventListenerImpl();
        singletonAppender.addLoggingEventListener(logEventListener);
    }

    public static class LogEventListenerImpl implements LogEventListener {

        @Override
        public void handle(LogEvent logEvent) {
            String msg = logEvent.getMessage();
            PrometheusMetricManager metrics = PrometheusMetricManager.getInstance();

            //TODO - add a method to parse message to an object
            if (msg.contains("event=Inference")) {
                String[] msgs = msg.split(",");
                metrics.incInferCount(msgs[0].split("=")[1], msgs[1].split("=")[1]);
            }else if(msg.contains("metric=InferLatency")){
                String[] msgs = msg.split(",");
                metrics.incInferLatency(Long.parseLong(msgs[3].split("=")[1]), msgs[0].split("=")[1], msgs[1].split("=")[1]);
            }else if(msg.contains("metric=QueueLatency")){
                String[] msgs = msg.split(",");
                metrics.incQueueLatency(Long.parseLong(msgs[3].split("=")[1]), msgs[0].split("=")[1], msgs[1].split("=")[1]);
            }
        }
    }
}



