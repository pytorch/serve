package org.pytorch.serve.grpcimpl;

import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.grpc.management.DescribeModelRequest;
import org.pytorch.serve.grpc.management.ListModelsRequest;
import org.pytorch.serve.grpc.management.ManagementAPIsServiceGrpc.ManagementAPIsServiceImplBase;
import org.pytorch.serve.grpc.management.ManagementResponse;
import org.pytorch.serve.grpc.management.RegisterModelRequest;
import org.pytorch.serve.grpc.management.ScaleWorkerRequest;
import org.pytorch.serve.grpc.management.SetDefaultRequest;
import org.pytorch.serve.grpc.management.UnregisterModelRequest;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.job.GRPCJob;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.GRPCUtils;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ManagementImpl extends ManagementAPIsServiceImplBase {
    private ConfigManager configManager;
    private static final Logger logger = LoggerFactory.getLogger(ManagementImpl.class);

    public ManagementImpl() {
        configManager = ConfigManager.getInstance();
    }

    @Override
    public void describeModel(
            DescribeModelRequest request, StreamObserver<ManagementResponse> responseObserver) {
        ((ServerCallStreamObserver<ManagementResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        String requestId = UUID.randomUUID().toString();
        RequestInput input = new RequestInput(requestId);
        String modelName = request.getModelName();
        String modelVersion = null;
        if (!request.getModelVersion().isEmpty()) {
            modelVersion = request.getModelVersion();
        }
        boolean customized = request.getCustomized();

        if ("all".equals(modelVersion) || !customized) {
            String resp;
            try {
                resp =
                        JsonUtils.GSON_PRETTY.toJson(
                                ApiUtils.getModelDescription(modelName, modelVersion));
                sendResponse(responseObserver, resp);
            } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
                sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
            }
        } else {
            input.updateHeaders("describe", "True");
            Job job = new GRPCJob(responseObserver, modelName, modelVersion, input);

            try {
                if (!ModelManager.getInstance().addJob(job)) {
                    String responseMessage = ApiUtils.getDescribeErrorResponseMessage(modelName);
                    InternalServerException e = new InternalServerException(responseMessage);
                    sendException(responseObserver, e, "InternalServerException.()");
                }
            } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
                sendErrorResponse(responseObserver, Status.INTERNAL, e);
            }
        }
    }

    @Override
    public void listModels(
            ListModelsRequest request, StreamObserver<ManagementResponse> responseObserver) {
        ((ServerCallStreamObserver<ManagementResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        int limit = request.getLimit();
        int pageToken = request.getNextPageToken();

        String modelList = JsonUtils.GSON_PRETTY.toJson(ApiUtils.getModelList(limit, pageToken));
        sendResponse(responseObserver, modelList);
    }

    @Override
    public void registerModel(
            RegisterModelRequest request, StreamObserver<ManagementResponse> responseObserver) {
        ((ServerCallStreamObserver<ManagementResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        org.pytorch.serve.http.messages.RegisterModelRequest registerModelRequest =
                new org.pytorch.serve.http.messages.RegisterModelRequest(request);

        StatusResponse statusResponse;
        try {
            if (!configManager.isModelApiEnabled()) {
                sendErrorResponse(
                        responseObserver,
                        Status.PERMISSION_DENIED,
                        new ModelException("Model API disabled"));
                return;
            }
            statusResponse = ApiUtils.registerModel(registerModelRequest);
            sendStatusResponse(responseObserver, statusResponse);
        } catch (InternalServerException e) {
            sendException(responseObserver, e, null);
        } catch (ExecutionException | InterruptedException | WorkerInitializationException e) {
            sendException(responseObserver, e, "Error while creating workers");
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
        } catch (ModelException | BadRequestException | DownloadArchiveException e) {
            sendErrorResponse(responseObserver, Status.INVALID_ARGUMENT, e);
        }
    }

    @Override
    public void scaleWorker(
            ScaleWorkerRequest request, StreamObserver<ManagementResponse> responseObserver) {
        ((ServerCallStreamObserver<ManagementResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        int minWorkers = GRPCUtils.getRegisterParam(request.getMinWorker(), 1);
        int maxWorkers = GRPCUtils.getRegisterParam(request.getMaxWorker(), minWorkers);
        String modelName = GRPCUtils.getRegisterParam(request.getModelName(), null);
        String modelVersion = GRPCUtils.getRegisterParam(request.getModelVersion(), null);
        boolean synchronous = request.getSynchronous();

        StatusResponse statusResponse;
        try {
            statusResponse =
                    ApiUtils.updateModelWorkers(
                            modelName,
                            modelVersion,
                            minWorkers,
                            maxWorkers,
                            synchronous,
                            false,
                            null);
            sendStatusResponse(responseObserver, statusResponse);
        } catch (ExecutionException | InterruptedException | WorkerInitializationException e) {
            sendException(responseObserver, e, "Error while creating workers");
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
        } catch (BadRequestException e) {
            sendErrorResponse(responseObserver, Status.INVALID_ARGUMENT, e);
        }
    }

    @Override
    public void setDefault(
            SetDefaultRequest request, StreamObserver<ManagementResponse> responseObserver) {
        ((ServerCallStreamObserver<ManagementResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        String modelName = request.getModelName();
        String newModelVersion = request.getModelVersion();

        try {
            String msg = ApiUtils.setDefault(modelName, newModelVersion);
            sendResponse(responseObserver, msg);
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
        }
    }

    @Override
    public void unregisterModel(
            UnregisterModelRequest request, StreamObserver<ManagementResponse> responseObserver) {
        ((ServerCallStreamObserver<ManagementResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        try {
            if (!configManager.isModelApiEnabled()) {
                sendErrorResponse(
                        responseObserver,
                        Status.PERMISSION_DENIED,
                        new ModelException("Model API disabled"));
                return;
            }
            String modelName = request.getModelName();
            if (modelName == null || ("").equals(modelName)) {
                sendErrorResponse(
                        responseObserver,
                        Status.INVALID_ARGUMENT,
                        new BadRequestException("Parameter url is required."));
            }

            String modelVersion = request.getModelVersion();

            if (("").equals(modelVersion)) {
                modelVersion = null;
            }
            ApiUtils.unregisterModel(modelName, modelVersion);
            String msg = "Model \"" + modelName + "\" unregistered";
            sendResponse(responseObserver, msg);
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
        } catch (BadRequestException e) {
            sendErrorResponse(responseObserver, Status.INVALID_ARGUMENT, e);
        }
    }

    private void sendResponse(StreamObserver<ManagementResponse> responseObserver, String msg) {
        ManagementResponse reply = ManagementResponse.newBuilder().setMsg(msg).build();
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }

    public static void sendErrorResponse(
            StreamObserver<ManagementResponse> responseObserver, Status status, Exception e) {
        responseObserver.onError(
                status.withDescription(e.getMessage())
                        .augmentDescription(e.getClass().getCanonicalName())
                        .asRuntimeException());
    }

    private void sendErrorResponse(
            StreamObserver<ManagementResponse> responseObserver,
            Status status,
            String description,
            String errorClass) {
        responseObserver.onError(
                status.withDescription(description)
                        .augmentDescription(errorClass)
                        .asRuntimeException());
    }

    private void sendStatusResponse(
            StreamObserver<ManagementResponse> responseObserver, StatusResponse statusResponse) {
        int httpResponseStatusCode = statusResponse.getHttpResponseCode();
        if (httpResponseStatusCode >= 200 && httpResponseStatusCode < 300) {
            sendResponse(responseObserver, statusResponse.getStatus());
        } else {
            sendErrorResponse(
                    responseObserver,
                    GRPCUtils.getGRPCStatusCode(statusResponse.getHttpResponseCode()),
                    statusResponse.getE().getMessage(),
                    statusResponse.getE().getClass().getCanonicalName());
        }
    }

    private void sendException(
            StreamObserver<ManagementResponse> responseObserver, Exception e, String description) {
        sendErrorResponse(
                responseObserver,
                Status.INTERNAL,
                description == null ? e.getMessage() : description,
                e.getClass().getCanonicalName());
    }
}
