package org.pytorch.serve.openapi;

import io.netty.handler.codec.http.HttpHeaderValues;
import io.prometheus.client.exporter.common.TextFormat;
import java.util.ArrayList;
import java.util.List;
import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.wlm.Model;

public final class OpenApiUtils {

    private OpenApiUtils() {}

    public static String listApis(ConnectorType type) {
        OpenApi openApi = new OpenApi();
        Info info = new Info();
        info.setTitle("TorchServe APIs");
        info.setDescription(
                "TorchServe is a flexible and easy to use tool for serving deep learning models");
        ConfigManager config = ConfigManager.getInstance();
        info.setVersion(config.getProperty("version", null));
        openApi.setInfo(info);

        if (ConnectorType.ALL.equals(type) || ConnectorType.INFERENCE_CONNECTOR.equals(type)) {
            listInferenceApis(openApi);
        }
        if (ConnectorType.ALL.equals(type) || ConnectorType.MANAGEMENT_CONNECTOR.equals(type)) {
            listManagementApis(openApi);
        }
        openApi.addPath("/metrics", getMetricsPath());
        return JsonUtils.GSON_PRETTY.toJson(openApi);
    }
    /**
     * The /v1/models/{model_name}:predict prediction api is used to access torchserve from
     * kfserving v1 predictor
     */
    private static void listInferenceApis(OpenApi openApi) {
        openApi.addPath("/", getApiDescriptionPath("apiDescription", false));
        openApi.addPath("/ping", getPingPath());
        openApi.addPath("/v1/models/{model_name}:predict", getPredictionsPath(false));
        openApi.addPath("/predictions/{model_name}", getPredictionsPath(false));
        openApi.addPath("/predictions/{model_name}/{model_version}", getPredictionsPath(true));
        openApi.addPath("/api-description", getApiDescriptionPath("api-description", true));
    }

    private static void listManagementApis(OpenApi openApi) {
        openApi.addPath("/", getApiDescriptionPath("apiDescription", false));
        openApi.addPath("/models", getModelsPath());
        openApi.addPath("/models/{model_name}", getModelManagerPath(false));
        openApi.addPath("/models/{model_name}/{model_version}", getModelManagerPath(true));
        openApi.addPath("/models/{model_name}/{model_version}/set-default", getSetDefaultPath());
        openApi.addPath("/api-description", getApiDescriptionPath("api-description", true));
    }

    public static String getModelApi(Model model) {
        String modelName = model.getModelName();
        OpenApi openApi = new OpenApi();
        Info info = new Info();
        info.setTitle("RESTful API for: " + modelName);
        info.setVersion("1.0.0");
        openApi.setInfo(info);

        openApi.addPath("/prediction/" + modelName, getModelPath(modelName));
        openApi.addPath("/v1/models/{model_name}:predict", getModelPath(modelName));

        return JsonUtils.GSON_PRETTY.toJson(openApi);
    }

    private static Path getApiDescriptionPath(String operationID, boolean legacy) {
        Schema schema = new Schema("object");
        schema.addProperty("openapi", new Schema("string"), true);
        schema.addProperty("info", new Schema("object"), true);
        schema.addProperty("paths", new Schema("object"), true);
        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Operation operation = new Operation(operationID, "Get openapi description.");
        operation.addResponse(new Response("200", "A openapi 3.0.1 descriptor", mediaType));
        operation.addResponse(new Response("500", "Internal Server Error", getErrorResponse()));

        Path path = new Path();
        if (legacy) {
            operation.setDeprecated(true);
            path.setGet(operation);
        } else {
            path.setOptions(operation);
        }
        return path;
    }

    private static Path getPingPath() {
        Schema schema = new Schema("object");
        schema.addProperty(
                "status", new Schema("string", "Overall status of the TorchServe."), true);
        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        Operation operation = new Operation("ping", "Get TorchServe status.");
        operation.addResponse(new Response("200", "TorchServe status", mediaType));
        operation.addResponse(new Response("500", "Internal Server Error", getErrorResponse()));

        Path path = new Path();
        path.setGet(operation);
        return path;
    }

    private static Path getPredictionsPath(boolean version) {
        String operationDescription;
        String operationId;
        if (version) {
            operationDescription =
                    "Predictions entry point to get inference using specific model version.";
            operationId = "version_predictions";
        } else {
            operationDescription =
                    "Predictions entry point to get inference using default model version.";
            operationId = "predictions";
        }
        Operation post = new Operation(operationId, operationDescription);
        post.addParameter(new PathParameter("model_name", "Name of model."));
        if (version) {
            post.addParameter(new PathParameter("model_version", "Name of model version."));
        }
        Schema schema = new Schema("string");
        schema.setFormat("binary");
        MediaType mediaType = new MediaType("*/*", schema);
        RequestBody requestBody = new RequestBody();
        requestBody.setDescription("Input data format is defined by each model.");
        requestBody.setRequired(true);
        requestBody.addContent(mediaType);

        post.setRequestBody(requestBody);

        schema = new Schema("string");
        schema.setFormat("binary");
        mediaType = new MediaType("*/*", schema);

        Response resp =
                new Response("200", "Output data format is defined by each model.", mediaType);
        post.addResponse(resp);

        MediaType error = getErrorResponse();
        post.addResponse(new Response("404", "Model not found or Model Version not found", error));
        post.addResponse(new Response("500", "Internal Server Error", error));
        post.addResponse(new Response("503", "No worker is available to serve request", error));

        Path path = new Path();
        path.setPost(post);
        return path;
    }

    private static Path getModelsPath() {
        Path path = new Path();
        path.setGet(getListModelsOperation());
        path.setPost(getRegisterOperation());
        return path;
    }

    private static Path getSetDefaultPath() {
        Path path = new Path();
        path.setPut(getSetDefaultOperation());
        return path;
    }

    private static Path getModelManagerPath(boolean version) {
        Path path = new Path();
        path.setGet(getDescribeModelOperation(version));
        path.setPut(getScaleOperation(version));
        path.setDelete(getUnRegisterOperation(version));
        return path;
    }

    private static Operation getSetDefaultOperation() {
        Operation operation = new Operation("setDefault", "Set default version of a model");
        operation.addParameter(
                new PathParameter(
                        "model_name", "Name of model whose default version needs to be updated."));
        operation.addParameter(
                new PathParameter(
                        "model_version",
                        "Version of model to be set as default version for the model"));

        MediaType status = getStatusResponse();
        MediaType error = getErrorResponse();

        operation.addResponse(
                new Response("200", "Default vesion succsesfully updated for model", status));
        operation.addResponse(
                new Response("404", "Model not found or Model version not found", error));
        operation.addResponse(new Response("500", "Internal Server Error", error));

        return operation;
    }

    private static Operation getListModelsOperation() {
        Operation operation = new Operation("listModels", "List registered models in TorchServe.");

        operation.addParameter(
                new QueryParameter(
                        "limit",
                        "integer",
                        "100",
                        "Use this parameter to specify the maximum number of items to return. When"
                                + " this value is present, TorchServe does not return more than the specified"
                                + " number of items, but it might return fewer. This value is optional. If you"
                                + " include a value, it must be between 1 and 1000, inclusive. If you do not"
                                + " include a value, it defaults to 100."));
        operation.addParameter(
                new QueryParameter(
                        "next_page_token",
                        "The token to retrieve the next set of results. TorchServe provides the"
                                + " token when the response from a previous call has more results than the"
                                + " maximum page size."));

        Schema schema = new Schema("object");
        schema.addProperty(
                "nextPageToken",
                new Schema(
                        "string",
                        "Use this parameter in a subsequent request after you receive a response"
                                + " with truncated results. Set it to the value of NextMarker from the"
                                + " truncated response you just received."),
                false);

        Schema modelProp = new Schema("object");
        modelProp.addProperty("modelName", new Schema("string", "Name of the model."), true);
        modelProp.addProperty("modelUrl", new Schema("string", "URL of the model."), true);
        Schema modelsProp = new Schema("array", "A list of registered models.");
        modelsProp.setItems(modelProp);
        schema.addProperty("models", modelsProp, true);
        MediaType json = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);

        operation.addResponse(new Response("200", "OK", json));
        operation.addResponse(new Response("500", "Internal Server Error", getErrorResponse()));
        return operation;
    }

    private static Operation getRegisterOperation() {
        Operation operation = new Operation("registerModel", "Register a new model in TorchServe.");

        operation.addParameter(
                new QueryParameter(
                        "url",
                        "string",
                        null,
                        true,
                        "Model archive download url, support local file or HTTP(s) protocol."
                                + " For S3, consider use pre-signed url."));
        operation.addParameter(
                new QueryParameter(
                        "model_name",
                        "Name of model. This value will override modelName in MANIFEST.json if present."));
        operation.addParameter(
                new QueryParameter(
                        "handler",
                        "Inference handler entry-point. This value will override handler in MANIFEST.json if present."));

        Parameter runtime =
                new QueryParameter(
                        "runtime",
                        "Runtime for the model custom service code. This value will override runtime in MANIFEST.json if present.");
        operation.addParameter(runtime);
        operation.addParameter(
                new QueryParameter(
                        "batch_size", "integer", "1", "Inference batch size, default: 1."));
        operation.addParameter(
                new QueryParameter(
                        "max_batch_delay",
                        "integer",
                        "100",
                        "Maximum delay for batch aggregation, default: 100."));
        operation.addParameter(
                new QueryParameter(
                        "response_timeout",
                        "integer",
                        "2",
                        "Maximum time, in seconds, the TorchServe waits for a response from the model inference code, default: 120."));
        operation.addParameter(
                new QueryParameter(
                        "initial_workers",
                        "integer",
                        "0",
                        "Number of initial workers, default: 0."));
        operation.addParameter(
                new QueryParameter(
                        "synchronous",
                        "boolean",
                        "false",
                        "Decides whether creation of worker synchronous or not, default: false."));

        Manifest.RuntimeType[] types = Manifest.RuntimeType.values();
        List<String> runtimeTypes = new ArrayList<>(types.length);
        for (Manifest.RuntimeType type : types) {
            runtimeTypes.add(type.toString());
        }
        runtime.getSchema().setEnumeration(runtimeTypes);

        MediaType status = getStatusResponse();
        MediaType error = getErrorResponse();

        operation.addResponse(new Response("200", "Model registered", status));
        operation.addResponse(new Response("202", "Accepted", status));
        operation.addResponse(new Response("210", "Partial Success", status));
        operation.addResponse(new Response("400", "Bad request", error));
        operation.addResponse(new Response("404", "Model not found", error));
        operation.addResponse(new Response("409", "Model already registered", error));
        operation.addResponse(new Response("500", "Internal Server Error", error));

        return operation;
    }

    private static Operation getUnRegisterOperation(boolean version) {
        String operationDescription;
        String operationId;
        if (version) {
            operationDescription =
                    "Unregister the specified version of a model from TorchServe. "
                            + "This is an asynchronous call by default. Caller can call listModels to confirm model is unregistered";
            operationId = "version_unregisterModel";
        } else {
            operationDescription =
                    "Unregister the default version of a model from TorchServe if it is the only version available."
                            + "This is a asynchronous call by default. Caller can call listModels to confirm model is unregistered";
            operationId = "unregisterModel";
        }
        Operation operation = new Operation(operationId, operationDescription);

        operation.addParameter(new PathParameter("model_name", "Name of model to unregister."));
        if (version) {
            operation.addParameter(
                    new PathParameter("model_version", "Version of model to unregister."));
        }
        operation.addParameter(
                new QueryParameter(
                        "synchronous",
                        "boolean",
                        "false",
                        "Decides whether the call is synchronous or not, default: false."));
        operation.addParameter(
                new QueryParameter(
                        "timeout",
                        "integer",
                        "-1",
                        "Waiting up to the specified wait time if necessary for"
                                + " a worker to complete all pending requests. Use 0 to terminate backend"
                                + " worker process immediately. Use -1 for wait infinitely."));

        MediaType status = getStatusResponse();
        MediaType error = getErrorResponse();

        operation.addResponse(new Response("200", "Model unregistered", status));
        operation.addResponse(new Response("202", "Accepted", status));
        operation.addResponse(
                new Response("404", "Model not found or Model version not found", error));
        operation.addResponse(new Response("408", "Request Timeout Error", error));
        operation.addResponse(new Response("500", "Internal Server Error", error));

        return operation;
    }

    private static Operation getDescribeModelOperation(boolean version) {
        String operationDescription;
        String operationId;
        if (version) {
            operationDescription =
                    "Provides detailed information about the specified version of a model."
                            + "If \"all\" is specified as version, returns the details about all the versions of the model.";
            operationId = "version_describeModel";
        } else {
            operationDescription =
                    "Provides detailed information about the default version of a model.";
            operationId = "describeModel";
        }

        Operation operation = new Operation(operationId, operationDescription);

        operation.addParameter(new PathParameter("model_name", "Name of model to describe."));
        if (version) {
            operation.addParameter(
                    new PathParameter("model_version", "Version of model to describe."));
        }
        Schema schema = new Schema("object");
        schema.addProperty("modelName", new Schema("string", "Name of the model."), true);
        schema.addProperty("modelVersion", new Schema("string", "Version of the model."), true);
        schema.addProperty("modelUrl", new Schema("string", "URL of the model."), true);
        schema.addProperty(
                "minWorkers", new Schema("integer", "Configured minimum number of worker."), true);
        schema.addProperty(
                "maxWorkers", new Schema("integer", "Configured maximum number of worker."), true);
        schema.addProperty("batchSize", new Schema("integer", "Configured batch size."), false);
        schema.addProperty(
                "maxBatchDelay",
                new Schema("integer", "Configured maximum batch delay in ms."),
                false);
        schema.addProperty(
                "status", new Schema("string", "Overall health status of the model"), true);

        Schema workers = new Schema("array", "A list of active backend workers.");
        Schema worker = new Schema("object");
        worker.addProperty("id", new Schema("string", "Worker id"), true);
        worker.addProperty("startTime", new Schema("string", "Worker start time"), true);
        worker.addProperty("gpu", new Schema("boolean", "If running on GPU"), false);
        Schema workerStatus = new Schema("string", "Worker status");
        List<String> status = new ArrayList<>();
        status.add("READY");
        status.add("LOADING");
        status.add("UNLOADING");
        workerStatus.setEnumeration(status);
        worker.addProperty("status", workerStatus, true);
        workers.setItems(worker);

        schema.addProperty("workers", workers, true);
        Schema metrics = new Schema("object");
        metrics.addProperty(
                "rejectedRequests",
                new Schema("integer", "Number requests has been rejected in last 10 minutes."),
                true);
        metrics.addProperty(
                "waitingQueueSize",
                new Schema("integer", "Number requests waiting in the queue."),
                true);
        metrics.addProperty(
                "requests",
                new Schema("integer", "Number requests processed in last 10 minutes."),
                true);
        schema.addProperty("metrics", metrics, true);

        MediaType mediaType = new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);
        MediaType error = getErrorResponse();

        operation.addResponse(new Response("200", "OK", mediaType));
        operation.addResponse(
                new Response("404", "Model not found or Model version not found", error));
        operation.addResponse(new Response("500", "Internal Server Error", error));

        return operation;
    }

    private static Operation getScaleOperation(boolean version) {
        String operationDescription;
        String operationId;
        if (version) {
            operationDescription =
                    "Configure number of workers for a specified version of a model. "
                            + "This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.";
            operationId = "version_setAutoScale";
        } else {
            operationDescription =
                    "Configure number of workers for a default version of a model."
                            + "This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.";
            operationId = "setAutoScale";
        }

        Operation operation = new Operation(operationId, operationDescription);
        operation.addParameter(new PathParameter("model_name", "Name of model to scale workers."));
        if (version) {
            operation.addParameter(
                    new PathParameter("model_version", "Version of model to scale workers."));
        }
        operation.addParameter(
                new QueryParameter(
                        "min_worker", "integer", "1", "Minimum number of worker processes."));
        operation.addParameter(
                new QueryParameter(
                        "max_worker", "integer", "1", "Maximum number of worker processes."));
        operation.addParameter(
                new QueryParameter(
                        "number_gpu", "integer", "0", "Number of GPU worker processes to create."));
        operation.addParameter(
                new QueryParameter(
                        "synchronous",
                        "boolean",
                        "false",
                        "Decides whether the call is synchronous or not, default: false."));
        operation.addParameter(
                new QueryParameter(
                        "timeout",
                        "integer",
                        "-1",
                        "Waiting up to the specified wait time if necessary for"
                                + " a worker to complete all pending requests. Use 0 to terminate backend"
                                + " worker process immediately. Use -1 for wait infinitely."));

        MediaType status = getStatusResponse();
        MediaType error = getErrorResponse();

        operation.addResponse(new Response("200", "Model workers updated", status));
        operation.addResponse(new Response("202", "Accepted", status));
        operation.addResponse(new Response("210", "Partial Success", status));
        operation.addResponse(new Response("400", "Bad request", error));
        operation.addResponse(
                new Response("404", "Model not found or Model version not found", error));
        operation.addResponse(new Response("500", "Internal Server Error", error));

        return operation;
    }

    private static Path getModelPath(String modelName) {
        Operation operation =
                new Operation(modelName, "A predict entry point for model: " + modelName + '.');
        operation.addResponse(new Response("200", "OK"));
        operation.addResponse(new Response("500", "Internal Server Error", getErrorResponse()));

        Path path = new Path();
        path.setPost(operation);
        return path;
    }

    private static Path getMetricsPath() {
        Schema schema = new Schema("object");
        schema.addProperty(
                "# HELP", new Schema("string", "Help text for TorchServe metric."), true);
        schema.addProperty("# TYPE", new Schema("string", "Type of TorchServe metric."), true);
        schema.addProperty("metric", new Schema("string", "TorchServe application metric."), true);
        MediaType mediaType = new MediaType(TextFormat.CONTENT_TYPE_004, schema);

        Operation operation =
                new Operation(
                        "metrics", "Get TorchServe application metrics in prometheus format.");
        operation.addParameter(new QueryParameter("name[]", "Names of metrics to filter"));
        operation.addResponse(new Response("200", "TorchServe application metrics", mediaType));
        operation.addResponse(new Response("500", "Internal Server Error", getErrorResponse()));

        Path path = new Path();
        path.setGet(operation);
        return path;
    }

    private static MediaType getErrorResponse() {
        Schema schema = new Schema("object");
        schema.addProperty("code", new Schema("integer", "Error code."), true);
        schema.addProperty("type", new Schema("string", "Error type."), true);
        schema.addProperty("message", new Schema("string", "Error message."), true);

        return new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);
    }

    private static MediaType getStatusResponse() {
        Schema schema = new Schema("object");
        schema.addProperty("status", new Schema("string", "Error type."), true);
        return new MediaType(HttpHeaderValues.APPLICATION_JSON.toString(), schema);
    }
}
