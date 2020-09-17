package org.pytorch.serve.job;

import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.Map;
import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.util.GRPCUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;

public class GRPCJob extends Job {
    private StreamObserver<PredictionResponse> responseObserver;

    public GRPCJob(
            StreamObserver<PredictionResponse> responseObserver,
            String modelName,
            String version,
            WorkerCommands cmd,
            RequestInput input) {
        super(modelName, version, cmd, input);
        this.responseObserver = responseObserver;
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
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }

    @Override
    public void sendError(HttpResponseStatus status, String error) {
        GRPCUtils.sendError(status.code(), error, responseObserver);
    }
}
