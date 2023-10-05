package org.pytorch.serve.job;

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

    private void cancelHandler(ServerCallStreamObserver<PredictionResponse> responseObserver) {
        if (responseObserver.isCancelled()) {
            logger.warn(
                    "grpc client call already cancelled, not able to send this response for requestId: {}",
                    getPayload().getRequestId());
        }
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
                cancelHandler(responseObserver);
                PredictionResponse reply =
                        PredictionResponse.newBuilder().setPrediction(output).build();
                responseObserver.onNext(reply);
                if (cmd == WorkerCommands.PREDICT
                        || (cmd == WorkerCommands.STREAMPREDICT
                                && responseHeaders
                                        .get(RequestInput.TS_STREAM_NEXT)
                                        .equals("false"))) {
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
                cancelHandler(responseObserver);
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
            default:
                break;
        }
    }

    @Override
    public boolean isOpen() {
        return ((ServerCallStreamObserver<PredictionResponse>) predictionResponseObserver)
                .isCancelled();
    }
}
