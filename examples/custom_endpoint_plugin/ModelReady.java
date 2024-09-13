package org.pytorch.serve.plugins.endpoint;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.pytorch.serve.servingsdk.Context;
import org.pytorch.serve.servingsdk.Model;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.Worker;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

@Endpoint(
        urlPattern = "model-ready",
        endpointType = EndpointTypes.INFERENCE,
        description = "Endpoint indicating registered model/s ready to serve inference requests")
public class ModelReady extends ModelServerEndpoint {
    private boolean modelsLoaded(Context ctx) {
        Map<String, Model> modelMap = ctx.getModels();

        if (modelMap.isEmpty()) {
            return false;
        }

        for (Map.Entry<String, Model> entry : modelMap.entrySet()) {
            boolean workerReady = false;
            for (Worker w : entry.getValue().getModelWorkers()) {
                if (w.isRunning()) {
                    workerReady = true;
                    break;
                }
            }
            if (!workerReady) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        if (modelsLoaded(ctx)) {
            rsp.setStatus(200, "Model/s ready");
            rsp.getOutputStream()
                    .write(
                            "{\n\t\"Status\": \"Model/s ready\"\n}\n"
                                    .getBytes(StandardCharsets.UTF_8));
        } else {
            rsp.setStatus(503, "Model/s not ready");
            rsp.getOutputStream()
                    .write(
                            "{\n\t\"Status\": \"Model/s not ready\"\n}\n"
                                    .getBytes(StandardCharsets.UTF_8));
        }
    }
}
