package org.pytorch.serve.grpcimpl;

import com.google.gson.Gson;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.grpc.openinference.GRPCInferenceServiceGrpc.GRPCInferenceServiceImplBase;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelInferRequest;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelInferRequest.InferInputTensor;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelInferResponse;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelMetadataRequest;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelMetadataResponse;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelMetadataResponse.TensorMetadata;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelReadyRequest;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ModelReadyResponse;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ServerLiveRequest;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ServerLiveResponse;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ServerReadyRequest;
import org.pytorch.serve.grpc.openinference.OpenInferenceGrpc.ServerReadyResponse;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.job.GRPCJob;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenInferenceProtocolImpl extends GRPCInferenceServiceImplBase {
    private static final Logger logger = LoggerFactory.getLogger(OpenInferenceProtocolImpl.class);

    @Override
    public void serverLive(
            ServerLiveRequest request, StreamObserver<ServerLiveResponse> responseObserver) {
        ((ServerCallStreamObserver<ServerLiveResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });

        ServerLiveResponse readyResponse = ServerLiveResponse.newBuilder().setLive(true).build();
        responseObserver.onNext(readyResponse);
        responseObserver.onCompleted();
    }

    @Override
    public void serverReady(
            ServerReadyRequest request, StreamObserver<ServerReadyResponse> responseObserver) {
        ((ServerCallStreamObserver<ServerReadyResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });

        ServerReadyResponse readyResponse = ServerReadyResponse.newBuilder().setReady(true).build();
        responseObserver.onNext(readyResponse);
        responseObserver.onCompleted();
    }

    private void sendErrorResponse(
            StreamObserver<?> responseObserver, Status internal, Exception e, String string) {
        responseObserver.onError(
                internal.withDescription(e.getMessage())
                        .augmentDescription(
                                string == null ? e.getClass().getCanonicalName() : string)
                        .withCause(e)
                        .asRuntimeException());
    }

    @Override
    public void modelReady(
            ModelReadyRequest request, StreamObserver<ModelReadyResponse> responseObserver) {
        ((ServerCallStreamObserver<ModelReadyResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        String modelName = request.getName();
        String modelVersion = request.getVersion();
        ModelManager modelManager = ModelManager.getInstance();
        boolean isModelReady = false;
        if (modelName == null || "".equals(modelName)) {
            BadRequestException e = new BadRequestException("Parameter name is required.");
            sendErrorResponse(responseObserver, Status.INTERNAL, e, "BadRequestException.()");
            return;
        }

        if (modelVersion == null || "".equals(modelVersion)) {
            modelVersion = null;
        }
        try {
            Model model = modelManager.getModel(modelName, modelVersion);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }

            int numScaled = model.getMinWorkers();
            int numHealthy = modelManager.getNumHealthyWorkers(model.getModelVersionName());
            isModelReady = numHealthy >= numScaled;

            ModelReadyResponse modelReadyResponse =
                    ModelReadyResponse.newBuilder().setReady(isModelReady).build();
            responseObserver.onNext(modelReadyResponse);
            responseObserver.onCompleted();

        } catch (ModelVersionNotFoundException | ModelNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e, null);
        }
    }

    @Override
    public void modelMetadata(
            ModelMetadataRequest request, StreamObserver<ModelMetadataResponse> responseObserver) {
        ((ServerCallStreamObserver<ModelMetadataResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        String modelName = request.getName();
        String modelVersion = request.getVersion();
        ModelManager modelManager = ModelManager.getInstance();
        ModelMetadataResponse.Builder response = ModelMetadataResponse.newBuilder();
        List<TensorMetadata> inputs = new ArrayList<>();
        List<TensorMetadata> outputs = new ArrayList<>();
        List<String> versions = new ArrayList<>();
        if (modelName == null || "".equals(modelName)) {
            BadRequestException e = new BadRequestException("Parameter model_name is required.");
            sendErrorResponse(responseObserver, Status.INTERNAL, e, "BadRequestException.()");
            return;
        }

        if (modelVersion == null || "".equals(modelVersion)) {
            modelVersion = null;
        }

        try {
            Model model = modelManager.getModel(modelName, modelVersion);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            modelManager
                    .getAllModelVersions(modelName)
                    .forEach(entry -> versions.add(entry.getKey()));
            response.setName(modelName);
            response.addAllVersions(versions);
            response.setPlatform("");
            response.addAllInputs(inputs);
            response.addAllOutputs(outputs);
            responseObserver.onNext(response.build());
            responseObserver.onCompleted();

        } catch (ModelVersionNotFoundException | ModelNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e, null);
        }
    }

    @Override
    public void modelInfer(
            ModelInferRequest request, StreamObserver<ModelInferResponse> responseObserver) {
        ((ServerCallStreamObserver<ModelInferResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });

        String modelName = request.getModelName();
        String modelVersion = request.getModelVersion();
        CharSequence contentsType = "application/json";
        Gson gson = new Gson();
        Map<String, Object> modelInferMap = new HashMap<>();
        List<Map<String, Object>> inferInputs = new ArrayList<>();
        String requestId = UUID.randomUUID().toString();
        RequestInput inputData = new RequestInput(requestId);

        // creating modelInfer map that same as kserve v2 existing request input data
        modelInferMap.put("id", request.getId());
        modelInferMap.put("model_name", request.getModelName());

        for (InferInputTensor entry : request.getInputsList()) {
            Map<String, Object> inferInputMap = new HashMap<>();
            inferInputMap.put("name", entry.getName());
            inferInputMap.put("shape", entry.getShapeList());
            inferInputMap.put("datatype", entry.getDatatype());
            setInputContents(entry, inferInputMap);
            inferInputs.add(inferInputMap);
        }
        modelInferMap.put("inputs", inferInputs);
        String jsonString = gson.toJson(modelInferMap);
        byte[] byteArray = jsonString.getBytes(StandardCharsets.UTF_8);

        if (modelName == null || "".equals(modelName)) {
            BadRequestException e = new BadRequestException("Parameter model_name is required.");
            sendErrorResponse(responseObserver, Status.INTERNAL, e, "BadRequestException.()");
            return;
        }

        if (modelVersion == null || "".equals(modelVersion)) {
            modelVersion = null;
        }

        try {
            ModelManager modelManager = ModelManager.getInstance();
            inputData.addParameter(new InputParameter("body", byteArray, contentsType));
            Job job =
                    new GRPCJob(
                            responseObserver,
                            modelName,
                            modelVersion,
                            inputData,
                            WorkerCommands.OIPPREDICT);

            if (!modelManager.addJob(job)) {
                String responseMessage =
                        ApiUtils.getStreamingInferenceErrorResponseMessage(modelName, modelVersion);
                InternalServerException e = new InternalServerException(responseMessage);
                sendErrorResponse(
                        responseObserver, Status.INTERNAL, e, "InternalServerException.()");
            }
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.INTERNAL, e, null);
        }
    }

    private static void setInputContents(
            InferInputTensor inferInputTensor, Map<String, Object> inferInputMap) {
        switch (inferInputTensor.getDatatype()) {
            case "BYTES":
                List<ByteString> byteStrings =
                        inferInputTensor.getContents().getBytesContentsList();
                List<String> base64Strings = new ArrayList<>();
                for (ByteString byteString : byteStrings) {
                    String base64String =
                            Base64.getEncoder().encodeToString(byteString.toByteArray());
                    base64Strings.add(base64String);
                }
                inferInputMap.put("data", base64Strings);
                break;

            case "FP32":
                List<Float> fp32Contents = inferInputTensor.getContents().getFp32ContentsList();
                inferInputMap.put("data", fp32Contents);
                break;

            case "FP64":
                List<Double> fp64ContentList = inferInputTensor.getContents().getFp64ContentsList();
                inferInputMap.put("data", fp64ContentList);
                break;

            case "INT8": // jump to INT32 case
            case "INT16": // jump to INT32 case
            case "INT32":
                List<Integer> int32Contents = inferInputTensor.getContents().getIntContentsList();
                inferInputMap.put("data", int32Contents);
                break;

            case "INT64":
                List<Long> int64Contents = inferInputTensor.getContents().getInt64ContentsList();
                inferInputMap.put("data", int64Contents);
                break;

            case "UINT8": // jump to UINT32 case
            case "UINT16": // jump to UINT32 case
            case "UINT32":
                List<Integer> uint32Contents = inferInputTensor.getContents().getUintContentsList();
                inferInputMap.put("data", uint32Contents);
                break;
            case "UINT64":
                List<Long> uint64Contents = inferInputTensor.getContents().getUint64ContentsList();
                inferInputMap.put("data", uint64Contents);
                break;

            case "BOOL":
                List<Boolean> boolContents = inferInputTensor.getContents().getBoolContentsList();
                inferInputMap.put("data", boolContents);
                break;

            default:
                break;
        }
    }
}
