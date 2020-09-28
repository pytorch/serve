package org.pytorch.serve.grpcimpl;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
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
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.GRPCUtils;
import org.pytorch.serve.util.JsonUtils;

public class ManagementImpl extends ManagementAPIsServiceImplBase {

    @Override
    public void describeModel(
            DescribeModelRequest request, StreamObserver<ManagementResponse> responseObserver) {

        String modelName = request.getModelName();
        String modelVersion = request.getModelVersion();

        String resp;
        try {
            resp =
                    JsonUtils.GSON_PRETTY.toJson(
                            ApiUtils.getModelDescription(modelName, modelVersion));
            sendResponse(responseObserver, resp);
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
        }
    }

    @Override
    public void listModels(
            ListModelsRequest request, StreamObserver<ManagementResponse> responseObserver) {
        int limit = request.getLimit();
        int pageToken = request.getNextPageToken();

        String modelList = JsonUtils.GSON_PRETTY.toJson(ApiUtils.getModelList(limit, pageToken));
        sendResponse(responseObserver, modelList);
    }

    @Override
    public void registerModel(
            RegisterModelRequest request, StreamObserver<ManagementResponse> responseObserver) {
        org.pytorch.serve.http.messages.RegisterModelRequest registerModelRequest =
                new org.pytorch.serve.http.messages.RegisterModelRequest(request);

        StatusResponse statusResponse;
        try {
            statusResponse = ApiUtils.registerModel(registerModelRequest);
            sendStatusResponse(responseObserver, statusResponse);
        } catch (InternalServerException e) {
            sendException(responseObserver, e, null);
        } catch (ExecutionException | InterruptedException e) {
            sendException(responseObserver, e, "Error while creating workers");
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.NOT_FOUND, e);
        } catch (ModelException | BadRequestException e) {
            sendErrorResponse(responseObserver, Status.INVALID_ARGUMENT, e);
        }
    }

    @Override
    public void scaleWorker(
            ScaleWorkerRequest request, StreamObserver<ManagementResponse> responseObserver) {
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
        } catch (ExecutionException | InterruptedException e) {
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
        try {
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

    private void sendErrorResponse(
            StreamObserver<ManagementResponse> responseObserver, Status status, Exception e) {
        responseObserver.onError(
                status.withDescription(e.getMessage())
                        .augmentDescription(e.getClass().getCanonicalName())
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
