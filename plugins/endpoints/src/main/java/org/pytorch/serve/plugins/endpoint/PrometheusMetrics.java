package org.pytorch.serve.plugins.endpoint;

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
        TextFormat.write004(writer, CollectorRegistry.defaultRegistry.filteredMetricFamilySamples(
                new HashSet<>(params)));


        // 6 * 1024 * 1024
        int maxRequestSize = Integer.parseInt(prop.getProperty("max_request_size", "6291456"));

        rsp.setContentType(TextFormat.CONTENT_TYPE_004);
        rsp.getOutputStream()
                .write(writer.toString().getBytes(StandardCharsets.UTF_8));
    }

}





