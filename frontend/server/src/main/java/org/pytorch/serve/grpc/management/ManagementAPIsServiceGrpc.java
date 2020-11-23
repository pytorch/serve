package org.pytorch.serve.grpc.management;

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
    comments = "Source: management.proto")
public final class ManagementAPIsServiceGrpc {

  private ManagementAPIsServiceGrpc() {}

  public static final String SERVICE_NAME = "org.pytorch.serve.grpc.management.ManagementAPIsService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.DescribeModelRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getDescribeModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "DescribeModel",
      requestType = org.pytorch.serve.grpc.management.DescribeModelRequest.class,
      responseType = org.pytorch.serve.grpc.management.ManagementResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.DescribeModelRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getDescribeModelMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.DescribeModelRequest, org.pytorch.serve.grpc.management.ManagementResponse> getDescribeModelMethod;
    if ((getDescribeModelMethod = ManagementAPIsServiceGrpc.getDescribeModelMethod) == null) {
      synchronized (ManagementAPIsServiceGrpc.class) {
        if ((getDescribeModelMethod = ManagementAPIsServiceGrpc.getDescribeModelMethod) == null) {
          ManagementAPIsServiceGrpc.getDescribeModelMethod = getDescribeModelMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.management.DescribeModelRequest, org.pytorch.serve.grpc.management.ManagementResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "DescribeModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.DescribeModelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ManagementResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ManagementAPIsServiceMethodDescriptorSupplier("DescribeModel"))
              .build();
        }
      }
    }
    return getDescribeModelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.ListModelsRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getListModelsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ListModels",
      requestType = org.pytorch.serve.grpc.management.ListModelsRequest.class,
      responseType = org.pytorch.serve.grpc.management.ManagementResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.ListModelsRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getListModelsMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.ListModelsRequest, org.pytorch.serve.grpc.management.ManagementResponse> getListModelsMethod;
    if ((getListModelsMethod = ManagementAPIsServiceGrpc.getListModelsMethod) == null) {
      synchronized (ManagementAPIsServiceGrpc.class) {
        if ((getListModelsMethod = ManagementAPIsServiceGrpc.getListModelsMethod) == null) {
          ManagementAPIsServiceGrpc.getListModelsMethod = getListModelsMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.management.ListModelsRequest, org.pytorch.serve.grpc.management.ManagementResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ListModels"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ListModelsRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ManagementResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ManagementAPIsServiceMethodDescriptorSupplier("ListModels"))
              .build();
        }
      }
    }
    return getListModelsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.RegisterModelRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getRegisterModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "RegisterModel",
      requestType = org.pytorch.serve.grpc.management.RegisterModelRequest.class,
      responseType = org.pytorch.serve.grpc.management.ManagementResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.RegisterModelRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getRegisterModelMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.RegisterModelRequest, org.pytorch.serve.grpc.management.ManagementResponse> getRegisterModelMethod;
    if ((getRegisterModelMethod = ManagementAPIsServiceGrpc.getRegisterModelMethod) == null) {
      synchronized (ManagementAPIsServiceGrpc.class) {
        if ((getRegisterModelMethod = ManagementAPIsServiceGrpc.getRegisterModelMethod) == null) {
          ManagementAPIsServiceGrpc.getRegisterModelMethod = getRegisterModelMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.management.RegisterModelRequest, org.pytorch.serve.grpc.management.ManagementResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "RegisterModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.RegisterModelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ManagementResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ManagementAPIsServiceMethodDescriptorSupplier("RegisterModel"))
              .build();
        }
      }
    }
    return getRegisterModelMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.ScaleWorkerRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getScaleWorkerMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ScaleWorker",
      requestType = org.pytorch.serve.grpc.management.ScaleWorkerRequest.class,
      responseType = org.pytorch.serve.grpc.management.ManagementResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.ScaleWorkerRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getScaleWorkerMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.ScaleWorkerRequest, org.pytorch.serve.grpc.management.ManagementResponse> getScaleWorkerMethod;
    if ((getScaleWorkerMethod = ManagementAPIsServiceGrpc.getScaleWorkerMethod) == null) {
      synchronized (ManagementAPIsServiceGrpc.class) {
        if ((getScaleWorkerMethod = ManagementAPIsServiceGrpc.getScaleWorkerMethod) == null) {
          ManagementAPIsServiceGrpc.getScaleWorkerMethod = getScaleWorkerMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.management.ScaleWorkerRequest, org.pytorch.serve.grpc.management.ManagementResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ScaleWorker"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ScaleWorkerRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ManagementResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ManagementAPIsServiceMethodDescriptorSupplier("ScaleWorker"))
              .build();
        }
      }
    }
    return getScaleWorkerMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.SetDefaultRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getSetDefaultMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "SetDefault",
      requestType = org.pytorch.serve.grpc.management.SetDefaultRequest.class,
      responseType = org.pytorch.serve.grpc.management.ManagementResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.SetDefaultRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getSetDefaultMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.SetDefaultRequest, org.pytorch.serve.grpc.management.ManagementResponse> getSetDefaultMethod;
    if ((getSetDefaultMethod = ManagementAPIsServiceGrpc.getSetDefaultMethod) == null) {
      synchronized (ManagementAPIsServiceGrpc.class) {
        if ((getSetDefaultMethod = ManagementAPIsServiceGrpc.getSetDefaultMethod) == null) {
          ManagementAPIsServiceGrpc.getSetDefaultMethod = getSetDefaultMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.management.SetDefaultRequest, org.pytorch.serve.grpc.management.ManagementResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "SetDefault"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.SetDefaultRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ManagementResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ManagementAPIsServiceMethodDescriptorSupplier("SetDefault"))
              .build();
        }
      }
    }
    return getSetDefaultMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.UnregisterModelRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getUnregisterModelMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "UnregisterModel",
      requestType = org.pytorch.serve.grpc.management.UnregisterModelRequest.class,
      responseType = org.pytorch.serve.grpc.management.ManagementResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.UnregisterModelRequest,
      org.pytorch.serve.grpc.management.ManagementResponse> getUnregisterModelMethod() {
    io.grpc.MethodDescriptor<org.pytorch.serve.grpc.management.UnregisterModelRequest, org.pytorch.serve.grpc.management.ManagementResponse> getUnregisterModelMethod;
    if ((getUnregisterModelMethod = ManagementAPIsServiceGrpc.getUnregisterModelMethod) == null) {
      synchronized (ManagementAPIsServiceGrpc.class) {
        if ((getUnregisterModelMethod = ManagementAPIsServiceGrpc.getUnregisterModelMethod) == null) {
          ManagementAPIsServiceGrpc.getUnregisterModelMethod = getUnregisterModelMethod =
              io.grpc.MethodDescriptor.<org.pytorch.serve.grpc.management.UnregisterModelRequest, org.pytorch.serve.grpc.management.ManagementResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "UnregisterModel"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.UnregisterModelRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.pytorch.serve.grpc.management.ManagementResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ManagementAPIsServiceMethodDescriptorSupplier("UnregisterModel"))
              .build();
        }
      }
    }
    return getUnregisterModelMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static ManagementAPIsServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ManagementAPIsServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ManagementAPIsServiceStub>() {
        @java.lang.Override
        public ManagementAPIsServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ManagementAPIsServiceStub(channel, callOptions);
        }
      };
    return ManagementAPIsServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static ManagementAPIsServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ManagementAPIsServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ManagementAPIsServiceBlockingStub>() {
        @java.lang.Override
        public ManagementAPIsServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ManagementAPIsServiceBlockingStub(channel, callOptions);
        }
      };
    return ManagementAPIsServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static ManagementAPIsServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ManagementAPIsServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ManagementAPIsServiceFutureStub>() {
        @java.lang.Override
        public ManagementAPIsServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ManagementAPIsServiceFutureStub(channel, callOptions);
        }
      };
    return ManagementAPIsServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public static abstract class ManagementAPIsServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     * Provides detailed information about the default version of a model.
     * </pre>
     */
    public void describeModel(org.pytorch.serve.grpc.management.DescribeModelRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getDescribeModelMethod(), responseObserver);
    }

    /**
     * <pre>
     * List registered models in TorchServe.
     * </pre>
     */
    public void listModels(org.pytorch.serve.grpc.management.ListModelsRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getListModelsMethod(), responseObserver);
    }

    /**
     * <pre>
     * Register a new model in TorchServe.
     * </pre>
     */
    public void registerModel(org.pytorch.serve.grpc.management.RegisterModelRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getRegisterModelMethod(), responseObserver);
    }

    /**
     * <pre>
     * Configure number of workers for a default version of a model.This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.
     * </pre>
     */
    public void scaleWorker(org.pytorch.serve.grpc.management.ScaleWorkerRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getScaleWorkerMethod(), responseObserver);
    }

    /**
     * <pre>
     * Set default version of a model
     * </pre>
     */
    public void setDefault(org.pytorch.serve.grpc.management.SetDefaultRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getSetDefaultMethod(), responseObserver);
    }

    /**
     * <pre>
     * Unregister the default version of a model from TorchServe if it is the only version available.This is a asynchronous call by default. Caller can call listModels to confirm model is unregistered
     * </pre>
     */
    public void unregisterModel(org.pytorch.serve.grpc.management.UnregisterModelRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getUnregisterModelMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getDescribeModelMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.management.DescribeModelRequest,
                org.pytorch.serve.grpc.management.ManagementResponse>(
                  this, METHODID_DESCRIBE_MODEL)))
          .addMethod(
            getListModelsMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.management.ListModelsRequest,
                org.pytorch.serve.grpc.management.ManagementResponse>(
                  this, METHODID_LIST_MODELS)))
          .addMethod(
            getRegisterModelMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.management.RegisterModelRequest,
                org.pytorch.serve.grpc.management.ManagementResponse>(
                  this, METHODID_REGISTER_MODEL)))
          .addMethod(
            getScaleWorkerMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.management.ScaleWorkerRequest,
                org.pytorch.serve.grpc.management.ManagementResponse>(
                  this, METHODID_SCALE_WORKER)))
          .addMethod(
            getSetDefaultMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.management.SetDefaultRequest,
                org.pytorch.serve.grpc.management.ManagementResponse>(
                  this, METHODID_SET_DEFAULT)))
          .addMethod(
            getUnregisterModelMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                org.pytorch.serve.grpc.management.UnregisterModelRequest,
                org.pytorch.serve.grpc.management.ManagementResponse>(
                  this, METHODID_UNREGISTER_MODEL)))
          .build();
    }
  }

  /**
   */
  public static final class ManagementAPIsServiceStub extends io.grpc.stub.AbstractAsyncStub<ManagementAPIsServiceStub> {
    private ManagementAPIsServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ManagementAPIsServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ManagementAPIsServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * Provides detailed information about the default version of a model.
     * </pre>
     */
    public void describeModel(org.pytorch.serve.grpc.management.DescribeModelRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getDescribeModelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * List registered models in TorchServe.
     * </pre>
     */
    public void listModels(org.pytorch.serve.grpc.management.ListModelsRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getListModelsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Register a new model in TorchServe.
     * </pre>
     */
    public void registerModel(org.pytorch.serve.grpc.management.RegisterModelRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getRegisterModelMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Configure number of workers for a default version of a model.This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.
     * </pre>
     */
    public void scaleWorker(org.pytorch.serve.grpc.management.ScaleWorkerRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getScaleWorkerMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Set default version of a model
     * </pre>
     */
    public void setDefault(org.pytorch.serve.grpc.management.SetDefaultRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSetDefaultMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * Unregister the default version of a model from TorchServe if it is the only version available.This is a asynchronous call by default. Caller can call listModels to confirm model is unregistered
     * </pre>
     */
    public void unregisterModel(org.pytorch.serve.grpc.management.UnregisterModelRequest request,
        io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getUnregisterModelMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   */
  public static final class ManagementAPIsServiceBlockingStub extends io.grpc.stub.AbstractBlockingStub<ManagementAPIsServiceBlockingStub> {
    private ManagementAPIsServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ManagementAPIsServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ManagementAPIsServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * Provides detailed information about the default version of a model.
     * </pre>
     */
    public org.pytorch.serve.grpc.management.ManagementResponse describeModel(org.pytorch.serve.grpc.management.DescribeModelRequest request) {
      return blockingUnaryCall(
          getChannel(), getDescribeModelMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * List registered models in TorchServe.
     * </pre>
     */
    public org.pytorch.serve.grpc.management.ManagementResponse listModels(org.pytorch.serve.grpc.management.ListModelsRequest request) {
      return blockingUnaryCall(
          getChannel(), getListModelsMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Register a new model in TorchServe.
     * </pre>
     */
    public org.pytorch.serve.grpc.management.ManagementResponse registerModel(org.pytorch.serve.grpc.management.RegisterModelRequest request) {
      return blockingUnaryCall(
          getChannel(), getRegisterModelMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Configure number of workers for a default version of a model.This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.
     * </pre>
     */
    public org.pytorch.serve.grpc.management.ManagementResponse scaleWorker(org.pytorch.serve.grpc.management.ScaleWorkerRequest request) {
      return blockingUnaryCall(
          getChannel(), getScaleWorkerMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Set default version of a model
     * </pre>
     */
    public org.pytorch.serve.grpc.management.ManagementResponse setDefault(org.pytorch.serve.grpc.management.SetDefaultRequest request) {
      return blockingUnaryCall(
          getChannel(), getSetDefaultMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * Unregister the default version of a model from TorchServe if it is the only version available.This is a asynchronous call by default. Caller can call listModels to confirm model is unregistered
     * </pre>
     */
    public org.pytorch.serve.grpc.management.ManagementResponse unregisterModel(org.pytorch.serve.grpc.management.UnregisterModelRequest request) {
      return blockingUnaryCall(
          getChannel(), getUnregisterModelMethod(), getCallOptions(), request);
    }
  }

  /**
   */
  public static final class ManagementAPIsServiceFutureStub extends io.grpc.stub.AbstractFutureStub<ManagementAPIsServiceFutureStub> {
    private ManagementAPIsServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ManagementAPIsServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ManagementAPIsServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * Provides detailed information about the default version of a model.
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.management.ManagementResponse> describeModel(
        org.pytorch.serve.grpc.management.DescribeModelRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getDescribeModelMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * List registered models in TorchServe.
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.management.ManagementResponse> listModels(
        org.pytorch.serve.grpc.management.ListModelsRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getListModelsMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Register a new model in TorchServe.
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.management.ManagementResponse> registerModel(
        org.pytorch.serve.grpc.management.RegisterModelRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getRegisterModelMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Configure number of workers for a default version of a model.This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.management.ManagementResponse> scaleWorker(
        org.pytorch.serve.grpc.management.ScaleWorkerRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getScaleWorkerMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Set default version of a model
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.management.ManagementResponse> setDefault(
        org.pytorch.serve.grpc.management.SetDefaultRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getSetDefaultMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * Unregister the default version of a model from TorchServe if it is the only version available.This is a asynchronous call by default. Caller can call listModels to confirm model is unregistered
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.pytorch.serve.grpc.management.ManagementResponse> unregisterModel(
        org.pytorch.serve.grpc.management.UnregisterModelRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getUnregisterModelMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_DESCRIBE_MODEL = 0;
  private static final int METHODID_LIST_MODELS = 1;
  private static final int METHODID_REGISTER_MODEL = 2;
  private static final int METHODID_SCALE_WORKER = 3;
  private static final int METHODID_SET_DEFAULT = 4;
  private static final int METHODID_UNREGISTER_MODEL = 5;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final ManagementAPIsServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(ManagementAPIsServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_DESCRIBE_MODEL:
          serviceImpl.describeModel((org.pytorch.serve.grpc.management.DescribeModelRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse>) responseObserver);
          break;
        case METHODID_LIST_MODELS:
          serviceImpl.listModels((org.pytorch.serve.grpc.management.ListModelsRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse>) responseObserver);
          break;
        case METHODID_REGISTER_MODEL:
          serviceImpl.registerModel((org.pytorch.serve.grpc.management.RegisterModelRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse>) responseObserver);
          break;
        case METHODID_SCALE_WORKER:
          serviceImpl.scaleWorker((org.pytorch.serve.grpc.management.ScaleWorkerRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse>) responseObserver);
          break;
        case METHODID_SET_DEFAULT:
          serviceImpl.setDefault((org.pytorch.serve.grpc.management.SetDefaultRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse>) responseObserver);
          break;
        case METHODID_UNREGISTER_MODEL:
          serviceImpl.unregisterModel((org.pytorch.serve.grpc.management.UnregisterModelRequest) request,
              (io.grpc.stub.StreamObserver<org.pytorch.serve.grpc.management.ManagementResponse>) responseObserver);
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

  private static abstract class ManagementAPIsServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    ManagementAPIsServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return org.pytorch.serve.grpc.management.Management.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("ManagementAPIsService");
    }
  }

  private static final class ManagementAPIsServiceFileDescriptorSupplier
      extends ManagementAPIsServiceBaseDescriptorSupplier {
    ManagementAPIsServiceFileDescriptorSupplier() {}
  }

  private static final class ManagementAPIsServiceMethodDescriptorSupplier
      extends ManagementAPIsServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    ManagementAPIsServiceMethodDescriptorSupplier(String methodName) {
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
      synchronized (ManagementAPIsServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new ManagementAPIsServiceFileDescriptorSupplier())
              .addMethod(getDescribeModelMethod())
              .addMethod(getListModelsMethod())
              .addMethod(getRegisterModelMethod())
              .addMethod(getScaleWorkerMethod())
              .addMethod(getSetDefaultMethod())
              .addMethod(getUnregisterModelMethod())
              .build();
        }
      }
    }
    return result;
  }
}
