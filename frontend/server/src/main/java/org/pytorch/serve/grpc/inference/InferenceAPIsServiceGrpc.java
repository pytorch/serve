package org.pytorch.serve.grpc.inference;

import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ClientCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ClientCalls.asyncClientStreamingCall;
import static io.grpc.stub.ClientCalls.asyncServerStreamingCall;
import static io.grpc.stub.ClientCalls.asyncUnaryCall;
import static io.grpc.stub.ClientCalls.blockingServerStreamingCall;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.stub.ServerCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ServerCalls.asyncClientStreamingCall;
import static io.grpc.stub.ServerCalls.asyncServerStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.31.1)",
    comments = "Source: inference.proto")
public final class InferenceAPIsServiceGrpc {

  private InferenceAPIsServiceGrpc() {}

  public static final String SERVICE_NAME = "org.pytorch.serve.grpc.inference.InferenceAPIsService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      org.pytorch.serve.grpc.inference.TorchServeHealthResponse> getPingMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Ping",
      requestType = com.google.protobuf.Empty.class,
      responseType = org.pytorch.serve.grpc.inference.TorchServeHealthResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.google.protobuf.Empty,
      org.pytorch.serve.grpc.inference.TorchServeHealthResponse> getPingMethod() {
    io.grpc.MethodDescriptor<com.google.protobuf.Empty, org.pytorch.serve.grpc.inference.TorchServeHealthResponse> getPingMethod;
    if ((getPingMethod = InferenceAPIsServiceGrpc.getPingMethod) == null) {
      synchronized (InferenceAPIsServiceGrpc.class) {
        if ((getPingMethod = InferenceAPIsServiceGrpc.getPingMethod) == null) {
          InferenceAPIsServiceGrpc.getPingMethod = getPingMethod =
              io.grpc.MethodDescriptor.<com.google.protobuf.Empty, org.pytorch.serve.grpc.inference.TorchServeHealthResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Ping"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.google.protobuf.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.inference.TorchServeHealthResponse.getDefaultInstance()))
              .setSchemaDescriptor(new InferenceAPIsServiceMethodDescriptorSupplier("Ping"))
              .build();
        }
      }
    }
    return getPingMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.inference.PredictionsRequest,
      org.pytorch.serve.grpc.inference.PredictionResponse> getPredictionsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "Predictions",
      requestType = org.pytorch.serve.grpc.inference.PredictionsRequest.class,
      responseType = org.pytorch.serve.grpc.inference.PredictionResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.inference.PredictionsRequest,
      org.pytorch.serve.grpc.inference.PredictionResponse> getPredictionsMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.inference.PredictionsRequest, org.pytorch.serve.grpc.inference.PredictionResponse> getPredictionsMethod;
    if ((getPredictionsMethod = InferenceAPIsServiceGrpc.getPredictionsMethod) == null) {
      synchronized (InferenceAPIsServiceGrpc.class) {
        if ((getPredictionsMethod = InferenceAPIsServiceGrpc.getPredictionsMethod) == null) {
          InferenceAPIsServiceGrpc.getPredictionsMethod = getPredictionsMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.inference.PredictionsRequest, org.pytorch.serve.grpc.inference.PredictionResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "Predictions"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.inference.PredictionsRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.inference.PredictionResponse.getDefaultInstance()))
              .setSchemaDescriptor(new InferenceAPIsServiceMethodDescriptorSupplier("Predictions"))
              .build();
        }
      }
    }
    return getPredictionsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static InferenceAPIsServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<InferenceAPIsServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<InferenceAPIsServiceStub>() {
        @java.lang.Override
        public InferenceAPIsServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new InferenceAPIsServiceStub(channel, callOptions);
        }
      };
    return InferenceAPIsServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static InferenceAPIsServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<InferenceAPIsServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<InferenceAPIsServiceBlockingStub>() {
        @java.lang.Override
        public InferenceAPIsServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new InferenceAPIsServiceBlockingStub(channel, callOptions);
        }
      };
    return InferenceAPIsServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static InferenceAPIsServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<InferenceAPIsServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<InferenceAPIsServiceFutureStub>() {
        @java.lang.Override
        public InferenceAPIsServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new InferenceAPIsServiceFutureStub(channel, callOptions);
        }
      };
    return InferenceAPIsServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class InferenceAPIsServiceImplBase implements io.grpc.BindableService {

    /**
     */
    public void ping(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.inference.TorchServeHealthResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getPingMethod(), responseObserver);
    }

    /**
     * <pre>
     * Predictions entry point to get inference using default model version.
     * </pre>
     */
    public void predictions(org.pytorch.serve.grpc.inference.PredictionsRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.inference.PredictionResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getPredictionsMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getPingMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                com.google.protobuf.Empty,
                org.pytorch.serve.grpc.inference.TorchServeHealthResponse>(
                  this, METHODID_PING)))
          .addMethod(
            getPredictionsMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.inference.PredictionsRequest,
                org.pytorch.serve.grpc.inference.PredictionResponse>(
                  this, METHODID_PREDICTIONS)))
          .build();
    }
  }

  /**
   */
  public static final class InferenceAPIsServiceStub extends io.grpc.stub.AbstractAsyncStub<InferenceAPIsServiceStub> {
    private InferenceAPIsServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected InferenceAPIsServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new InferenceAPIsServiceStub(channel, callOptions);
    }

    /**
     */
    public void ping(com.google.protobuf.Empty request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.inference.TorchServeHealthResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getPingMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Predictions entry point to get inference using default model version.
     * </pre>
     */
    public void predictions(org.pytorch.serve.grpc.inference.PredictionsRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.inference.PredictionResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getPredictionsMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class InferenceAPIsServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<InferenceAPIsServiceBlockingStub> {
    private InferenceAPIsServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected InferenceAPIsServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new InferenceAPIsServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public org.pytorch.serve.grpc.inference.TorchServeHealthResponse ping(com.google.protobuf.Empty request) {
      return blockingUnaryCall(
          getChannel(), getPingMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Predictions entry point to get inference using default model version.
     * </pre>
     */
    public org.pytorch.serve.grpc.inference.PredictionResponse predictions(org.pytorch.serve.grpc.inference.PredictionsRequest request) {
      return blockingUnaryCall(
          getChannel(), getPredictionsMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class InferenceAPIsServiceFutureStub extends io.grpc.stub.AbstractFutureStub<InferenceAPIsServiceFutureStub> {
    private InferenceAPIsServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected InferenceAPIsServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new InferenceAPIsServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.inference.TorchServeHealthResponse> ping(
        com.google.protobuf.Empty request) {
      return futureUnaryCall(
          getChannel().newCall(getPingMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Predictions entry point to get inference using default model version.
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.inference.PredictionResponse> predictions(
        org.pytorch.serve.grpc.inference.PredictionsRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getPredictionsMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_PING = 0;
  private static final int METHODID_PREDICTIONS = 1;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final InferenceAPIsServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(InferenceAPIsServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_PING:
          serviceImpl.ping((com.google.protobuf.Empty) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.inference.TorchServeHealthResponse>) responseObserver);
          break;
        case METHODID_PREDICTIONS:
          serviceImpl.predictions((org.pytorch.serve.grpc.inference.PredictionsRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.inference.PredictionResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class InferenceAPIsServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    InferenceAPIsServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return org.pytorch.serve.grpc.inference.Inference.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("InferenceAPIsService");
    }
  }

  private static final class InferenceAPIsServiceFileDescriptorSupplier
      extends InferenceAPIsServiceBaseDescriptorSupplier {
    InferenceAPIsServiceFileDescriptorSupplier() {}
  }

  private static final class InferenceAPIsServiceMethodDescriptorSupplier
      extends InferenceAPIsServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    InferenceAPIsServiceMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (InferenceAPIsServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new InferenceAPIsServiceFileDescriptorSupplier())
              .addMethod(getPingMethod())
              .addMethod(getPredictionsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
