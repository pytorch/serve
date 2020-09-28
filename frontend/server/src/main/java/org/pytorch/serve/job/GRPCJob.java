package org.pytorch.serve.job;

import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import java.util.Map;
import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;

public class GRPCJob extends Job {
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
    }

    @Override
    public void sendError(int status, String error) {
        predictionResponseObserver.onError(new InternalServerException(error));
    }
}
