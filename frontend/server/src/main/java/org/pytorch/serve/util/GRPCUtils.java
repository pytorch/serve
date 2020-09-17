package org.pytorch.serve.util;

import io.grpc.stub.StreamObserver;
import org.pytorch.serve.grpc.inference.PredictionResponse;

public final class GRPCUtils {

    private GRPCUtils() {}

    public static void sendError(
            int status, String error, StreamObserver<PredictionResponse> responseObserver) {
        PredictionResponse reply =
                PredictionResponse.newBuilder().setStatusCode(status).setInfo(error).build();
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }
}
