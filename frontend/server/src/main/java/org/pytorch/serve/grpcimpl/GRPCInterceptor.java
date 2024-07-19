package org.pytorch.serve.grpcimpl;

import io.grpc.ForwardingServerCall;
import io.grpc.Grpc;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import org.pytorch.serve.http.Session;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.TokenAuthorization;
import org.pytorch.serve.util.TokenAuthorization.TokenType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GRPCInterceptor implements ServerInterceptor {

    private TokenType tokenType;
    private static final Metadata.Key<String> tokenAuthHeaderKey =
            Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER);
    private static final Logger logger = LoggerFactory.getLogger("ACCESS_LOG");

    public GRPCInterceptor(ConnectorType connectorType) {
        switch (connectorType) {
            case MANAGEMENT_CONNECTOR:
                tokenType = TokenType.MANAGEMENT;
                break;
            case INFERENCE_CONNECTOR:
                tokenType = TokenType.INFERENCE;
                break;
            default:
                tokenType = TokenType.INFERENCE;
        }
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
        String inetSocketString =
                call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
        String serviceName = call.getMethodDescriptor().getFullMethodName();
        Session session = new Session(inetSocketString, serviceName);

        if (TokenAuthorization.isEnabled()) {
            if (!headers.containsKey(tokenAuthHeaderKey)
                    || !checkTokenAuthorization(headers.get(tokenAuthHeaderKey))) {
                call.close(
                        Status.PERMISSION_DENIED.withDescription(
                                "Token Authorization failed. Token either incorrect, expired, or not provided correctly"),
                        new Metadata());
                return new ServerCall.Listener<ReqT>() {};
            }
        }

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

    private Boolean checkTokenAuthorization(String tokenAuthHeaderValue) {
        if (tokenAuthHeaderValue == null) {
            return false;
        }
        String token = TokenAuthorization.parseTokenFromBearerTokenHeader(tokenAuthHeaderValue);

        return TokenAuthorization.checkTokenAuthorization(token, tokenType);
    }
}
