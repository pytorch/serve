package org.pytorch.serve.grpcimpl;

import io.grpc.ForwardingServerCall;
import io.grpc.Grpc;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import org.pytorch.serve.http.Session;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GRPCInterceptor implements ServerInterceptor {

    private static final Logger logger = LoggerFactory.getLogger("ACCESS_LOG");

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
        String inetSocketString =
                call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
        String serviceName = call.getMethodDescriptor().getFullMethodName();
        Session session = new Session(inetSocketString, serviceName);

        return next.startCall(
                new ForwardingServerCall.SimpleForwardingServerCall<ReqT, RespT>(call) {
                    @Override
                    public void close(final Status status, final Metadata trailers) {
                        session.setCode(status.getCode().value());
                        logger.info(session.toString());
                        super.close(status, trailers);
                    }
                },
                headers);
    }
}
