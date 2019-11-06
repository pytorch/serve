package software.amazon.ai.mms.plugins.endpoint;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import software.amazon.ai.mms.servingsdk.Context;
import software.amazon.ai.mms.servingsdk.Model;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;
import software.amazon.ai.mms.servingsdk.Worker;
import software.amazon.ai.mms.servingsdk.annotations.Endpoint;
import software.amazon.ai.mms.servingsdk.annotations.helpers.EndpointTypes;
import software.amazon.ai.mms.servingsdk.http.Request;
import software.amazon.ai.mms.servingsdk.http.Response;

@Endpoint(
        urlPattern = "ping",
        endpointType = EndpointTypes.INFERENCE,
        description = "Ping endpoint for sagemaker containers.")
public class Ping extends ModelServerEndpoint {
    private boolean init;
    private byte[] success = "{\n\t\"Status\": \"Healthy\"\n}\n".getBytes(StandardCharsets.UTF_8);

    private boolean modelsLoaded(Context ctx) {
        Map<String, Model> modelMap = ctx.getModels();

        for (Map.Entry<String, Model> entry : modelMap.entrySet()) {
            for (Worker w : entry.getValue().getModelWorkers()) {
                if (w.isRunning()) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean validConfig(String svc) {
        String fileName = svc;
        if (svc.contains(":")) {
            fileName = svc.substring(0, svc.lastIndexOf(':'));
        }
        if (!fileName.contains(".py")) {
            fileName = fileName.concat(".py");
        }
        return new File(fileName).exists();
    }

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        rsp.setStatus(200);
        String isMultiModelMode = System.getenv("SAGEMAKER_MULTI_MODE");
        if (isMultiModelMode == null || "false".equalsIgnoreCase(isMultiModelMode)) {
            if (!init && !modelsLoaded(ctx)) {
                rsp.setStatus(503, "Model loading...");
                rsp.getOutputStream()
                        .write("Models are not loaded".getBytes(StandardCharsets.UTF_8));
            } else {
                init = true;
                rsp.getOutputStream().write(success);
            }
        } else {
            String svcFile = ctx.getConfig().getProperty("default_service_handler");
            if ((svcFile == null) || !validConfig(svcFile)) {
                rsp.setStatus(503, "Service file unavailable");
                rsp.getOutputStream()
                        .write("Service file unavailable".getBytes(StandardCharsets.UTF_8));
            } else {
                rsp.getOutputStream().write(success);
            }
        }
    }
}
