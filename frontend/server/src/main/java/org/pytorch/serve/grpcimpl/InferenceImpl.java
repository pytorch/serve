package org.pytorch.serve.grpcimpl;

import com.google.protobuf.ByteString;
import com.google.protobuf.Empty;
import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.net.HttpURLConnection;
import java.util.Arrays;
import java.util.List;
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
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InferenceImpl extends InferenceAPIsServiceImplBase {
    private static final Logger logger = LoggerFactory.getLogger(InferenceImpl.class);

    @Override
    public void ping(Empty request, StreamObserver<TorchServeHealthResponse> responseObserver) {
        ((ServerCallStreamObserver<TorchServeHealthResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    io.grpc.Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        Runnable r =
                () -> {
                    boolean isHealthy = ApiUtils.isModelHealthy();
                    int code = HttpURLConnection.HTTP_OK;
                    String response = "Healthy";
                    if (!isHealthy) {
                        response = "Unhealthy";
                        code = HttpURLConnection.HTTP_INTERNAL_ERROR;
                    }

                    TorchServeHealthResponse reply =
                            TorchServeHealthResponse.newBuilder()
                                    .setHealth(
                                            JsonUtils.GSON_PRETTY_EXPOSED.toJson(
                                                    new StatusResponse(response, code)))
                                    .build();
                    responseObserver.onNext(reply);
                    responseObserver.onCompleted();
                };
        ApiUtils.getTorchServeHealth(r);
    }

    @Override
    public void predictions(
            PredictionsRequest request, StreamObserver<PredictionResponse> responseObserver) {
        prediction(request, responseObserver, WorkerCommands.PREDICT);
    }

    @Override
    public void streamPredictions(
            PredictionsRequest request, StreamObserver<PredictionResponse> responseObserver) {
        logger.info("streamPredictions get req");
        prediction(request, responseObserver, WorkerCommands.STREAMPREDICT);
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

    private void prediction(
            PredictionsRequest request,
            StreamObserver<PredictionResponse> responseObserver,
            WorkerCommands workerCmd) {
        ((ServerCallStreamObserver<PredictionResponse>) responseObserver)
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
        try {
            ModelManager modelManager = ModelManager.getInstance();
            Model model = modelManager.getModel(modelName, modelVersion);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            inputData.setClientExpireTS(model.getClientTimeoutInMills());

            for (Map.Entry<String, ByteString> entry : request.getInputMap().entrySet()) {
                inputData.addParameter(
                        new InputParameter(entry.getKey(), entry.getValue().toByteArray()));
            }

            IMetric inferenceRequestsTotalMetric =
                    MetricCache.getInstance().getMetricFrontend("ts_inference_requests_total");
            if (inferenceRequestsTotalMetric != null) {
                List<String> inferenceRequestsTotalMetricDimensionValues =
                        Arrays.asList(
                                modelName,
                                modelVersion == null ? "default" : modelVersion,
                                ConfigManager.getInstance().getHostName());
                try {
                    inferenceRequestsTotalMetric.addOrUpdate(
                            inferenceRequestsTotalMetricDimensionValues, 1);
                } catch (Exception e) {
                    logger.error(
                            "Failed to update frontend metric ts_inference_requests_total: ", e);
                }
            }

            Job job = new GRPCJob(responseObserver, modelName, modelVersion, workerCmd, inputData);

            if (!modelManager.addJob(job)) {
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
}
