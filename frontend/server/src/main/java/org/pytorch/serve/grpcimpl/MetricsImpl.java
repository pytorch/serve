package org.pytorch.serve.grpcimpl;

import io.grpc.stub.StreamObserver;
import org.pytorch.serve.grpc.metrics.MetricsAPIsServiceGrpc.MetricsAPIsServiceImplBase;
import org.pytorch.serve.grpc.metrics.MetricsRequest;
import org.pytorch.serve.grpc.metrics.MetricsResponse;

public class MetricsImpl extends MetricsAPIsServiceImplBase {

    @Override
    public void metrics(MetricsRequest request, StreamObserver<MetricsResponse> responseObserver) {
        super.metrics(request, responseObserver);
    }
}
