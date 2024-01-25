package org.pytorch.serve.grpcimpl;

import io.grpc.BindableService;
import org.pytorch.serve.util.ConnectorType;

public final class GRPCServiceFactory {

    private GRPCServiceFactory() {}

    public static BindableService getgRPCService(ConnectorType connectorType) {
        BindableService torchServeService = null;
        switch (connectorType) {
            case MANAGEMENT_CONNECTOR:
                torchServeService = new ManagementImpl();
                break;
            case INFERENCE_CONNECTOR:
                torchServeService = new InferenceImpl();
                break;
            case OPEN_INFERENCE_CONNECTOR:
                torchServeService = new OpenInferenceProtocolImpl();
                break;
            default:
                break;
        }
        return torchServeService;
    }
}
