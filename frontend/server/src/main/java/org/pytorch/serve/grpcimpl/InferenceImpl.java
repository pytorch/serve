package org.pytorch.serve.grpcimpl;

import com.google.protobuf.ByteString;
import com.google.protobuf.Empty;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.net.HttpURLConnection;
import java.util.Map;
import java.util.UUID;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
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

        if (modelName == null || ("").equals(modelName)) {
            BadRequestException e = new BadRequestException("Parameter model_name is required.");
            responseObserver.onError(
                    Status.INTERNAL
                            .withDescription(e.getMessage())
                            .augmentDescription("BadRequestException.()")
                            .withCause(e)
                            .asRuntimeException());
            return;
        }

        if (modelVersion == null || ("").equals(modelVersion)) {
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
                        "Model \""
                                + modelName
                                + "\" Version "
                                + modelVersion
                                + " has no worker to serve inference request. Please use scale workers API to add workers.";

                if (modelVersion == null) {
                    responseMessage =
                            "Model \""
                                    + modelName
                                    + "\" has no worker to serve inference request. Please use scale workers API to add workers.";
                }
                InternalServerException e = new InternalServerException(responseMessage);
                responseObserver.onError(
                        Status.INTERNAL
                                .withDescription(e.getMessage())
                                .augmentDescription("InternalServerException.()")
                                .asRuntimeException());
            }
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            responseObserver.onError(
                    Status.INTERNAL
                            .withDescription(e.getMessage())
                            .augmentDescription(e.getClass().getCanonicalName())
                            .asRuntimeException());
        }
    }
}
