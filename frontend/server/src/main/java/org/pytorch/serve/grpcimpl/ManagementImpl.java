package org.pytorch.serve.grpcimpl;

import io.grpc.stub.StreamObserver;
import org.pytorch.serve.grpc.management.DescribeModelRequest;
import org.pytorch.serve.grpc.management.DescribeModelResponse;
import org.pytorch.serve.grpc.management.ListModelsRequest;
import org.pytorch.serve.grpc.management.ListModelsResponse;
import org.pytorch.serve.grpc.management.ManagementAPIsServiceGrpc.ManagementAPIsServiceImplBase;
import org.pytorch.serve.grpc.management.RegisterModelRequest;
import org.pytorch.serve.grpc.management.RegisterModelResponse;
import org.pytorch.serve.grpc.management.ScaleWorkerRequest;
import org.pytorch.serve.grpc.management.ScaleWorkerResponse;
import org.pytorch.serve.grpc.management.SetDefaultRequest;
import org.pytorch.serve.grpc.management.SetDefaultResponse;
import org.pytorch.serve.grpc.management.UnregisterModelRequest;
import org.pytorch.serve.grpc.management.UnregisterModelResponse;

public class ManagementImpl extends ManagementAPIsServiceImplBase {

    @Override
    public void describeModel(
            DescribeModelRequest request, StreamObserver<DescribeModelResponse> responseObserver) {
        super.describeModel(request, responseObserver);
    }

    @Override
    public void listModels(
            ListModelsRequest request, StreamObserver<ListModelsResponse> responseObserver) {
        super.listModels(request, responseObserver);
    }

    @Override
    public void registerModel(
            RegisterModelRequest request, StreamObserver<RegisterModelResponse> responseObserver) {
        super.registerModel(request, responseObserver);
    }

    @Override
    public void scaleWorker(
            ScaleWorkerRequest request, StreamObserver<ScaleWorkerResponse> responseObserver) {
        super.scaleWorker(request, responseObserver);
    }

    @Override
    public void setDefault(
            SetDefaultRequest request, StreamObserver<SetDefaultResponse> responseObserver) {
        super.setDefault(request, responseObserver);
    }

    @Override
    public void unregisterModel(
            UnregisterModelRequest request,
            StreamObserver<UnregisterModelResponse> responseObserver) {
        super.unregisterModel(request, responseObserver);
    }
}
