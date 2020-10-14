package org.pytorch.serve.test.plugins.metrics;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.servingsdk.Context;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

@Endpoint(
        urlPattern = "metrics",
        endpointType = EndpointTypes.METRIC,
        description = "Test Metric endpoint")
public class MetricsEndpoint extends ModelServerEndpoint {

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        Map<String, List<String>> paramsMap = req.getParameterMap();
        List<String> params = paramsMap.getOrDefault("name[]", Collections.emptyList());
        TestMetricManager metricManager = TestMetricManager.getInstance();
        HashMap<String, String> data = metricManager.getData();
        Gson gson = new Gson();
        String json = gson.toJson(data);
        rsp.getOutputStream().write(json.getBytes(StandardCharsets.UTF_8));
    }
}
