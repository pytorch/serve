package org.pytorch.serve.plugins.endpoint.prometheus;

import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.exporter.common.TextFormat;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.pytorch.serve.servingsdk.Context;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

/**
 * This class extends ModelServerEndpoint from Torch Serve SDK and acts as Prometheus Metric
 * endpoint At the time of initialization of Torch Serve server, the class gets loaded.
 */
@Endpoint(
        urlPattern = "metrics",
        endpointType = EndpointTypes.METRIC,
        description = "Prometheus Metric endpoint")
public class PrometheusMetrics extends ModelServerEndpoint {

    /** Handle the Get Request and respond back with requested Prometheus Metrics */
    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        Map<String, List<String>> params_map = req.getParameterMap();
        List<String> params = params_map.getOrDefault("name[]", Collections.emptyList());

        Writer writer = new StringWriter();
        TextFormat.write004(
                writer,
                CollectorRegistry.defaultRegistry.filteredMetricFamilySamples(
                        new HashSet<>(params)));

        rsp.setContentType(TextFormat.CONTENT_TYPE_004);
        rsp.getOutputStream().write(writer.toString().getBytes(StandardCharsets.UTF_8));
    }
}
