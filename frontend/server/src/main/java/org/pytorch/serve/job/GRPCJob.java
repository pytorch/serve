package org.pytorch.serve.job;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.rpc.ErrorInfo;
import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.grpc.management.ManagementResponse;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.InferTensorContents;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelInferResponse;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelInferResponse.InferOutputTensor;
import org.pytorch.serve.grpcimpl.ManagementImpl;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.GRPCUtils;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GRPCJob extends Job {
    private static final Logger logger = LoggerFactory.getLogger(GRPCJob.class);

    private final IMetric queueTimeMetric;
    private final List<String> queueTimeMetricDimensionValues;
    private StreamObserver<PredictionResponse> predictionResponseObserver;
    private StreamObserver<ManagementResponse> managementResponseObserver;
    private StreamObserver<ModelInferResponse> modelInferResponseObserver;

    public GRPCJob(
            StreamObserver<PredictionResponse> predictionResponseObserver,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        super(modelName, version, cmd, input);
        this.predictionResponseObserver = predictionResponseObserver;
        this.queueTimeMetric = MetricCache.getInstance().getMetricFrontend("QueueTime");
        this.queueTimeMetricDimensionValues =
                Arrays.asList("Host", ConfigManager.getInstance().getHostName());
    }

    public GRPCJob(
            StreamObserver<ModelInferResponse> modelInferResponseObserver,
            String modelName,
            String version,
            RequestInput input,
            WorkerCommands cmd) {
        super(modelName, version, cmd, input);
        this.modelInferResponseObserver = modelInferResponseObserver;
        this.queueTimeMetric = MetricCache.getInstance().getMetricFrontend("QueueTime");
        this.queueTimeMetricDimensionValues =
                Arrays.asList("Host", ConfigManager.getInstance().getHostName());
    }

    public GRPCJob(
            StreamObserver<ManagementResponse> managementResponseObserver,
            String modelName,
            String version,
            RequestInput input) {
        super(modelName, version, WorkerCommands.DESCRIBE, input);
        this.managementResponseObserver = managementResponseObserver;
        this.queueTimeMetric = MetricCache.getInstance().getMetricFrontend("QueueTime");
        this.queueTimeMetricDimensionValues =
                Arrays.asList("Host", ConfigManager.getInstance().getHostName());
    }

    private boolean cancelHandler(ServerCallStreamObserver<PredictionResponse> responseObserver) {
        if (responseObserver.isCancelled()) {
            logger.warn(
                    "grpc client call already cancelled, not able to send this response for requestId: {}",
                    getPayload().getRequestId());
            return true;
        }
        return false;
    }

    private void logQueueTime() {
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
                this.queueTimeMetric.addOrUpdate(this.queueTimeMetricDimensionValues, queueTime);
            } catch (Exception e) {
                logger.error("Failed to update frontend metric QueueTime: ", e);
            }
        }
    }

    @Override
    public void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {
        ByteString output = ByteString.copyFrom(body);
        WorkerCommands cmd = this.getCmd();

        switch (cmd) {
            case PREDICT:
            case STREAMPREDICT:
            case STREAMPREDICT2:
                ServerCallStreamObserver<PredictionResponse> responseObserver =
                        (ServerCallStreamObserver<PredictionResponse>) predictionResponseObserver;
                if (cancelHandler(responseObserver)) {
                    // issue #3087: Leave response early as the request has been canceled.
                    // Note: trying to continue wil trigger an exception when calling `onNext`.
                    return;
                }
                PredictionResponse reply =
                        PredictionResponse.newBuilder().setPrediction(output).build();
                responseObserver.onNext(reply);
                if (cmd == WorkerCommands.PREDICT
                        || (cmd == WorkerCommands.STREAMPREDICT
                                && responseHeaders
                                        .get(RequestInput.TS_STREAM_NEXT)
                                        .equals("false"))) {
                    if (cancelHandler(responseObserver)) {
                        return;
                    }
                    responseObserver.onCompleted();
                    logQueueTime();
                } else if (cmd == WorkerCommands.STREAMPREDICT2
                        && (responseHeaders.get(RequestInput.TS_STREAM_NEXT) == null
                                || responseHeaders
                                        .get(RequestInput.TS_STREAM_NEXT)
                                        .equals("false"))) {
                    logQueueTime();
                }
                break;
            case DESCRIBE:
                try {
                    ArrayList<DescribeModelResponse> respList =
                            ApiUtils.getModelDescription(
                                    this.getModelName(), this.getModelVersion());
                    if (!output.isEmpty() && respList != null && respList.size() == 1) {
                        respList.get(0).setCustomizedMetadata(body);
                    }
                    String resp = JsonUtils.GSON_PRETTY.toJson(respList);
                    ManagementResponse mgmtReply =
                            ManagementResponse.newBuilder().setMsg(resp).build();
                    managementResponseObserver.onNext(mgmtReply);
                    managementResponseObserver.onCompleted();
                } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
                    ManagementImpl.sendErrorResponse(
                            managementResponseObserver, Status.NOT_FOUND, e);
                }
                break;
            case OIPPREDICT:
                Gson gson = new Gson();
                String jsonResponse = output.toStringUtf8();
                JsonObject jsonObject = gson.fromJson(jsonResponse, JsonObject.class);
                if (((ServerCallStreamObserver<ModelInferResponse>) modelInferResponseObserver)
                        .isCancelled()) {
                    logger.warn(
                            "grpc client call already cancelled, not able to send this response for requestId: {}",
                            getPayload().getRequestId());
                    return;
                }
                ModelInferResponse.Builder responseBuilder = ModelInferResponse.newBuilder();
                responseBuilder.setId(jsonObject.get("id").getAsString());
                responseBuilder.setModelName(jsonObject.get("model_name").getAsString());
                responseBuilder.setModelVersion(jsonObject.get("model_version").getAsString());
                JsonArray jsonOutputs = jsonObject.get("outputs").getAsJsonArray();

                for (JsonElement element : jsonOutputs) {
                    InferOutputTensor.Builder outputBuilder = InferOutputTensor.newBuilder();
                    outputBuilder.setName(element.getAsJsonObject().get("name").getAsString());
                    outputBuilder.setDatatype(
                            element.getAsJsonObject().get("datatype").getAsString());
                    JsonArray shapeArray = element.getAsJsonObject().get("shape").getAsJsonArray();
                    shapeArray.forEach(
                            shapeElement -> outputBuilder.addShape(shapeElement.getAsLong()));
                    setOutputContents(element, outputBuilder);
                    responseBuilder.addOutputs(outputBuilder);
                }
                modelInferResponseObserver.onNext(responseBuilder.build());
                modelInferResponseObserver.onCompleted();
                break;
            default:
                break;
        }
    }

    @Override
    public void sendError(int status, String error) {
        Status responseStatus = GRPCUtils.getGRPCStatusCode(status);
        WorkerCommands cmd = this.getCmd();

        switch (cmd) {
            case PREDICT:
            case STREAMPREDICT:
            case STREAMPREDICT2:
                ServerCallStreamObserver<PredictionResponse> responseObserver =
                        (ServerCallStreamObserver<PredictionResponse>) predictionResponseObserver;
                if (cancelHandler(responseObserver)) {
                    // issue #3087: Leave response early as the request has been canceled.
                    // Note: trying to continue wil trigger an exception when calling `onNext`.
                    return;
                }
                if (cmd == WorkerCommands.PREDICT || cmd == WorkerCommands.STREAMPREDICT) {
                    responseObserver.onError(
                            responseStatus
                                    .withDescription(error)
                                    .augmentDescription(
                                            "org.pytorch.serve.http.InternalServerException")
                                    .asRuntimeException());
                } else if (cmd == WorkerCommands.STREAMPREDICT2) {
                    com.google.rpc.Status rpcStatus =
                            com.google.rpc.Status.newBuilder()
                                    .setCode(responseStatus.getCode().value())
                                    .setMessage(error)
                                    .addDetails(
                                            Any.pack(
                                                    ErrorInfo.newBuilder()
                                                            .setReason(
                                                                    "org.pytorch.serve.http.InternalServerException")
                                                            .build()))
                                    .build();
                    responseObserver.onNext(
                            PredictionResponse.newBuilder()
                                    .setPrediction(null)
                                    .setStatus(rpcStatus)
                                    .build());
                }
                break;
            case DESCRIBE:
                managementResponseObserver.onError(
                        responseStatus
                                .withDescription(error)
                                .augmentDescription(
                                        "org.pytorch.serve.http.InternalServerException")
                                .asRuntimeException());
                break;
            case OIPPREDICT:
                modelInferResponseObserver.onError(
                        responseStatus
                                .withDescription(error)
                                .augmentDescription(
                                        "org.pytorch.serve.http.InternalServerException")
                                .asRuntimeException());
                break;
            default:
                break;
        }
    }

    @Override
    public boolean isOpen() {
        return ((ServerCallStreamObserver<PredictionResponse>) predictionResponseObserver)
                .isCancelled();
    }

    private void setOutputContents(JsonElement element, InferOutputTensor.Builder outputBuilder) {
        String dataType = element.getAsJsonObject().get("datatype").getAsString();
        JsonArray jsonData = element.getAsJsonObject().get("data").getAsJsonArray();
        InferTensorContents.Builder inferTensorContents = InferTensorContents.newBuilder();
        switch (dataType) {
            case "INT8": // jump to INT32 case
            case "INT16": // jump to INT32 case
            case "INT32": // intContents
                List<Integer> int32Contents = new ArrayList<>();
                jsonData.forEach(data -> int32Contents.add(data.getAsInt()));
                inferTensorContents.addAllIntContents(int32Contents);
                break;

            case "INT64": // int64Contents
                List<Long> int64Contents = new ArrayList<>();
                jsonData.forEach(data -> int64Contents.add(data.getAsLong()));
                inferTensorContents.addAllInt64Contents(int64Contents);
                break;

            case "BYTES": // bytesContents
                List<ByteString> byteContents = new ArrayList<>();
                jsonData.forEach(
                        data -> byteContents.add(ByteString.copyFromUtf8(data.toString())));
                inferTensorContents.addAllBytesContents(byteContents);
                break;

            case "BOOL": // boolContents
                List<Boolean> boolContents = new ArrayList<>();
                jsonData.forEach(data -> boolContents.add(data.getAsBoolean()));
                inferTensorContents.addAllBoolContents(boolContents);
                break;

            case "FP32": // fp32Contents
                List<Float> fp32Contents = new ArrayList<>();
                jsonData.forEach(data -> fp32Contents.add(data.getAsFloat()));
                inferTensorContents.addAllFp32Contents(fp32Contents);
                break;

            case "FP64": // fp64Contents
                List<Double> fp64Contents = new ArrayList<>();
                jsonData.forEach(data -> fp64Contents.add(data.getAsDouble()));
                inferTensorContents.addAllFp64Contents(fp64Contents);
                break;

            case "UINT8": // jump to UINT32 case
            case "UINT16": // jump to UINT32 case
            case "UINT32": // uint32Contents
                List<Integer> uint32Contents = new ArrayList<>();
                jsonData.forEach(data -> uint32Contents.add(data.getAsInt()));
                inferTensorContents.addAllUintContents(uint32Contents);
                break;

            case "UINT64": // uint64Contents
                List<Long> uint64Contents = new ArrayList<>();
                jsonData.forEach(data -> uint64Contents.add(data.getAsLong()));
                inferTensorContents.addAllUint64Contents(uint64Contents);
                break;
            default:
                break;
        }
        outputBuilder.setContents(inferTensorContents); // set output contents
    }
}
