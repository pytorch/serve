package org.pytorch.serve.grpcimpl;

import com.google.protobuf.ByteString;
import com.google.protobuf.Empty;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.net.HttpURLConnection;
import java.util.Map;
import java.util.UUID;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.grpc.inference.InferenceAPIsServiceGrpc.InferenceAPIsServiceImplBase;
import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.grpc.inference.PredictionsRequest;
import org.pytorch.serve.grpc.inference.TorchServeHealthResponse;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.job.GRPCJob;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.metrics.api.MetricAggregator;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.ModelManager;

public class InferenceImpl extends InferenceAPIsServiceImplBase {

    @Override
    public void ping(Empty request, StreamObserver<TorchServeHealthResponse> responseObserver) {
        Runnable r =
                () -> {
                    String response = ApiUtils.getWorkerStatus();
                    TorchServeHealthResponse reply =
                            TorchServeHealthResponse.newBuilder()
                                    .setHealth(
                                            JsonUtils.GSON_PRETTY_EXPOSED.toJson(
                                                    new StatusResponse(
                                                            response, HttpURLConnection.HTTP_OK)))
                                    .build();
                    responseObserver.onNext(reply);
                    responseObserver.onCompleted();
                };
        ApiUtils.getTorchServeHealth(r);
    }

    @Override
    public void predictions(
            PredictionsRequest request, StreamObserver<PredictionResponse> responseObserver) {
        String modelName = request.getModelName();
        String modelVersion = request.getModelVersion();

        if (modelName == null || "".equals(modelName)) {
            BadRequestException e = new BadRequestException("Parameter model_name is required.");
            sendErrorResponse(responseObserver, Status.INTERNAL, e, "BadRequestException.()");
            return;
        }

        if (modelVersion == null || "".equals(modelVersion)) {
            modelVersion = null;
        }

        String requestId = UUID.randomUUID().toString();
        RequestInput inputData = new RequestInput(requestId);

        for (Map.Entry<String, ByteString> entry : request.getInputMap().entrySet()) {
            inputData.addParameter(
                    new InputParameter(entry.getKey(), entry.getValue().toByteArray()));
        }

        MetricAggregator.handleInferenceMetric(modelName, modelVersion);
        Job job =
                new GRPCJob(
                        responseObserver,
                        modelName,
                        modelVersion,
                        WorkerCommands.PREDICT,
                        inputData);

        try {
            if (!ModelManager.getInstance().addJob(job)) {
                String responseMessage =
                        ApiUtils.getInferenceErrorResponseMessage(modelName, modelVersion);
                InternalServerException e = new InternalServerException(responseMessage);
                sendErrorResponse(
                        responseObserver, Status.INTERNAL, e, "InternalServerException.()");
            }
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            sendErrorResponse(responseObserver, Status.INTERNAL, e, null);
        }
    }

    private void sendErrorResponse(
            StreamObserver<PredictionResponse> responseObserver,
            Status status,
            Exception e,
            String description) {
        responseObserver.onError(
                status.withDescription(e.getMessage())
                        .augmentDescription(
                                description == null ? e.getClass().getCanonicalName() : description)
                        .withCause(e)
                        .asRuntimeException());
    }
}
