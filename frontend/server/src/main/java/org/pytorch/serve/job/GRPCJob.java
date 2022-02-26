package org.pytorch.serve.job;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.grpc.management.ManagementResponse;
import org.pytorch.serve.grpcimpl.ManagementImpl;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.GRPCUtils;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GRPCJob extends Job {
    private static final Logger logger = LoggerFactory.getLogger(Job.class);
    private static final Logger loggerTsMetrics =
            LoggerFactory.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);
    private static final Dimension DIMENSION = new Dimension("Level", "Host");

    private StreamObserver<PredictionResponse> predictionResponseObserver;
    private StreamObserver<ManagementResponse> managementResponseObserver;

    public GRPCJob(
            StreamObserver<PredictionResponse> predictionResponseObserver,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        super(modelName, version, cmd, input);
        this.predictionResponseObserver = predictionResponseObserver;
    }

    public GRPCJob(
            StreamObserver<ManagementResponse> managementResponseObserver,
            String modelName,
            String version,
            RequestInput input) {
        super(modelName, version, WorkerCommands.DESCRIBE, input);
        this.managementResponseObserver = managementResponseObserver;
    }

    @Override
    public void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {

        ByteString output = ByteString.copyFrom(body);
        if (this.getCmd() == WorkerCommands.PREDICT) {
            PredictionResponse reply =
                    PredictionResponse.newBuilder().setPrediction(output).build();
            predictionResponseObserver.onNext(reply);
            predictionResponseObserver.onCompleted();

            logger.debug(
                    "Waiting time ns: {}, Backend time ns: {}",
                    getScheduled() - getBegin(),
                    System.nanoTime() - getScheduled());
            String queueTime =
                    String.valueOf(
                            TimeUnit.MILLISECONDS.convert(
                                    getScheduled() - getBegin(), TimeUnit.NANOSECONDS));
            loggerTsMetrics.info(
                    "{}",
                    new Metric(
                            "QueueTime",
                            queueTime,
                            "ms",
                            ConfigManager.getInstance().getHostName(),
                            DIMENSION));
        } else if (this.getCmd() == WorkerCommands.DESCRIBE) {
            try {
                ArrayList<DescribeModelResponse> respList =
                        ApiUtils.getModelDescription(this.getModelName(), this.getModelVersion());
                if (!output.isEmpty() && respList != null && respList.size() == 1) {
                    respList.get(0).setCustomizedMetadata(body);
                }
                String resp = JsonUtils.GSON_PRETTY.toJson(respList);
                ManagementResponse reply = ManagementResponse.newBuilder().setMsg(resp).build();
                managementResponseObserver.onNext(reply);
                managementResponseObserver.onCompleted();
            } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
                ManagementImpl.sendErrorResponse(managementResponseObserver, Status.NOT_FOUND, e);
            }
        }
    }

    @Override
    public void sendError(int status, String error) {
        Status responseStatus = GRPCUtils.getGRPCStatusCode(status);
        if (this.getCmd() == WorkerCommands.PREDICT) {
            predictionResponseObserver.onError(
                    responseStatus
                            .withDescription(error)
                            .augmentDescription("org.pytorch.serve.http.InternalServerException")
                            .asRuntimeException());
        } else if (this.getCmd() == WorkerCommands.DESCRIBE) {
            managementResponseObserver.onError(
                    responseStatus
                            .withDescription(error)
                            .augmentDescription("org.pytorch.serve.http.InternalServerException")
                            .asRuntimeException());
        }
    }
}
