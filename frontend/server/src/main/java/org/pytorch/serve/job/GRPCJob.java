package org.pytorch.serve.job;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.GRPCUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GRPCJob extends Job {
    private static final Logger logger = LoggerFactory.getLogger(Job.class);
    private static final org.apache.log4j.Logger loggerTsMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);
    private static final Dimension DIMENSION = new Dimension("Level", "Host");

    private StreamObserver<PredictionResponse> predictionResponseObserver;

    public GRPCJob(
            StreamObserver<PredictionResponse> predictionResponseObserver,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        super(modelName, version, cmd, input);
        this.predictionResponseObserver = predictionResponseObserver;
    }

    @Override
    public void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders) {

        ByteString output = ByteString.copyFrom(body);
        PredictionResponse reply = PredictionResponse.newBuilder().setPrediction(output).build();
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
                new Metric(
                        "QueueTime",
                        queueTime,
                        "ms",
                        ConfigManager.getInstance().getHostName(),
                        DIMENSION));
    }

    @Override
    public void sendError(int status, String error) {
        Status responseStatus = GRPCUtils.getGRPCStatusCode(status);
        predictionResponseObserver.onError(
                responseStatus
                        .withDescription(error)
                        .augmentDescription("org.pytorch.serve.http.InternalServerException")
                        .asRuntimeException());
    }
}
